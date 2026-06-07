#!/usr/bin/env python3
"""Replay the mixed drifting trace against a vLLM endpoint (Initial Plan 2 → step 3).

Fires the requests built by ``make_trace.py`` at an OpenAI-compatible vLLM server
(``/v1/completions``) at a controlled arrival rate, in trace order, so the domain segments
play out over wall-clock while the fork's ``expert_load_logger`` captures per-window expert
load. Emits two streams into the run dir:

  requests.jsonl  — per request: req_id, segment_label, t_start, t_end, latency, tokens, ok
  segments.jsonl  — one row per segment block with its measured [t_start, t_end] span and
                    request count → the timeline ``simulate``/``charts`` use to label expert
                    windows by domain (Q6) via ``telemetry.assign_segments``.

Dependency-light: stdlib ``urllib`` + a thread pool. ``--rate`` sets a Poisson-ish open-loop
arrival rate (req/s); ``--concurrency`` caps in-flight requests. With ``--dry-run`` it does
not hit the network — it just writes the segment timeline at the scheduled cadence (useful
for wiring tests). Wall-clock here is ``time.time()`` so it joins the hook's window stamps.

Usage:
  python -m b2tel.replay --trace traces/mixed.jsonl --url http://127.0.0.1:8000 \
      --model allenai/OLMoE-1B-7B-0924-Instruct --rate 8 --concurrency 16 \
      --out /u3/u3anand/b2/runs/capture/<ts>
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from b2tel.telemetry import JsonlWriter


def _post(url: str, payload: dict, timeout: float) -> tuple[bool, int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return True, resp.status, body
    except urllib.error.HTTPError as e:
        return False, e.code, {}
    except Exception:  # noqa: BLE001 (connection refused, timeout, etc.)
        return False, 0, {}


def load_trace(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay the B2 trace against vLLM.")
    ap.add_argument("--trace", required=True, help="trace .jsonl from make_trace")
    ap.add_argument("--url", default="http://127.0.0.1:8000",
                    help="vLLM base URL (':8000' shorthand also accepted)")
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924-Instruct")
    ap.add_argument("--rate", type=float, default=8.0, help="arrival rate (req/s); 0 = closed-loop")
    ap.add_argument("--concurrency", type=int, default=16, help="max in-flight requests")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-tokens", type=int, default=0, help="override per-request cap (0 = use trace)")
    ap.add_argument("--limit", type=int, default=0, help="replay only first N requests (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="don't hit the network; emit timeline only")
    ap.add_argument("--out", required=True, help="run dir for requests.jsonl + segments.jsonl")
    args = ap.parse_args(argv)

    base = args.url
    if base.startswith(":"):
        base = "http://127.0.0.1" + base
    base = base.rstrip("/")
    comp_url = base + "/v1/completions"

    trace = load_trace(args.trace)
    if args.limit:
        trace = trace[: args.limit]
    if not trace:
        print("[replay] empty trace", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    req_log = JsonlWriter(out_dir / "requests.jsonl")
    seg_log = JsonlWriter(out_dir / "segments.jsonl")

    # segment timeline accumulator (first/last request wall time per block)
    seg_lock = threading.Lock()
    seg_spans: dict[int, dict] = {}  # block_idx -> {label, t_start, t_end, n}

    def note_segment(block_idx: int, label: str, t0: float, t1: float) -> None:
        with seg_lock:
            s = seg_spans.get(block_idx)
            if s is None:
                seg_spans[block_idx] = {"segment_label": label, "t_start": t0, "t_end": t1, "n": 1}
            else:
                s["t_start"] = min(s["t_start"], t0)
                s["t_end"] = max(s["t_end"], t1)
                s["n"] += 1

    def fire(rec: dict) -> None:
        t0 = time.time()
        ok, status, body = (True, 200, {})
        ptoks = ctoks = 0
        if not args.dry_run:
            payload = {
                "model": args.model,
                "prompt": rec["prompt"],
                "max_tokens": args.max_tokens or rec.get("max_tokens", 256),
                "temperature": 0.0,
            }
            ok, status, body = _post(comp_url, payload, args.timeout)
            usage = (body or {}).get("usage") or {}
            ptoks = usage.get("prompt_tokens", 0)
            ctoks = usage.get("completion_tokens", 0)
        t1 = time.time()
        note_segment(rec["block_idx"], rec["segment_label"], t0, t1)
        req_log.write(
            req_id=rec["req_id"], segment_label=rec["segment_label"],
            t_start=t0, t_end=t1, latency_s=t1 - t0,
            prompt_tokens=ptoks, completion_tokens=ctoks, ok=ok, status=status,
        )

    interval = (1.0 / args.rate) if args.rate and args.rate > 0 else 0.0
    t_run0 = time.time()
    print(f"[replay] {len(trace)} reqs → {comp_url} rate={args.rate}/s "
          f"conc={args.concurrency} dry={args.dry_run}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = []
        for i, rec in enumerate(trace):
            if interval:  # open-loop: pace dispatch to target arrival rate
                target = t_run0 + i * interval
                slack = target - time.time()
                if slack > 0:
                    time.sleep(slack)
            futures.append(pool.submit(fire, rec))
        for f in futures:
            f.result()

    # flush segment timeline (sorted by block order)
    n_ok = 0
    for block_idx in sorted(seg_spans):
        s = seg_spans[block_idx]
        seg_log.write(
            segment_label=s["segment_label"], t_start=s["t_start"],
            t_end=s["t_end"], n_requests=s["n"], block_idx=block_idx,
        )
        n_ok += s["n"]
    req_log.close()
    seg_log.close()
    dt = time.time() - t_run0
    print(f"[replay] done: {n_ok} requests in {dt:.1f}s "
          f"({n_ok / dt:.1f} req/s) → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
