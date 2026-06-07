#!/usr/bin/env python3
"""Migration micro-bench — the Q3 kill-test (step 5).

Two measurements joined into one verdict:

1. **Migration cost** (needs 2 GPUs): the bytes of one OLMoE expert's weights and the time
   to move them across PCIe between the fast (Ada/L40S) and slow (A6000) tiers — no NVLink
   on watgpu308, so this is a Gen4 PCIe copy (peer-to-peer if enabled, else staged through
   host). Reported as ms/expert and effective GB/s.

2. **Hot-set turnover period** (GPU-free; reads the B3 capture ``expert.jsonl``): how long a
   layer's hot expert set stays put before it churns (Jaccard < threshold).

**Verdict (Q3):** migrating an expert is viable *iff* ``migration_ms`` ≪ ``turnover_period``
— there's time to move a hot expert onto the fast tier before the hot set moves on. If not,
TierShift must fall back to **replication-only** (keep replicas resident, never migrate
live). This single number shapes the controller.

Output ``migration_cost.json``::

    {
      "expert_bytes": ..., "dtype": "bfloat16",
      "pcie": {"ms_per_expert": ..., "GBps": ..., "mode": "peer"|"host-staged"},
      "turnover": {"median_s": ..., "mean_s": ..., "censored": ...},
      "verdict": {"migration_ms": ..., "turnover_period_s": ..., "migrate_viable": bool,
                  "headroom_x": turnover/migration}
    }

Usage:
  python -m b2tel.migrate_bench --fast 6 --slow 4 \
      --capture /u3/u3anand/b2/runs/capture/<ts>/expert.jsonl \
      --out /u3/u3anand/b2/runs/migration_cost.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HIDDEN = 2048
INTERMEDIATE = 1024
BF16_BYTES = 2


def expert_param_bytes(hidden: int, inter: int, dbytes: int) -> int:
    # gate + up: hidden*inter each; down: inter*hidden  → 3 * hidden * inter params
    return 3 * hidden * inter * dbytes


def _measure_pcie(fast: int, slow: int, nbytes: int, iters: int, warmup: int) -> dict:
    import torch

    src_dev = torch.device(f"cuda:{slow}")  # hot expert currently lives on the slow tier
    dst_dev = torch.device(f"cuda:{fast}")  # migrate onto the fast tier
    n = nbytes // 2  # bf16 elements
    src = torch.empty(n, dtype=torch.bfloat16, device=src_dev)
    dst = torch.empty(n, dtype=torch.bfloat16, device=dst_dev)

    # try enabling peer access; fall back to host-staged copy
    mode = "peer"
    try:
        dst.copy_(src)  # direct device-to-device (peer or implicit host stage)
    except Exception:  # noqa: BLE001
        mode = "host-staged"

    def one_copy():
        if mode == "peer":
            dst.copy_(src)
        else:
            dst.copy_(src.to("cpu").to(dst_dev))

    for _ in range(warmup):
        one_copy()
    torch.cuda.synchronize(src_dev)
    torch.cuda.synchronize(dst_dev)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream(dst_dev))
    for _ in range(iters):
        one_copy()
    end.record(torch.cuda.current_stream(dst_dev))
    torch.cuda.synchronize(dst_dev)
    ms = start.elapsed_time(end) / iters
    gbps = (nbytes / 1e9) / (ms / 1e3) if ms > 0 else 0.0
    return {"ms_per_expert": ms, "GBps": gbps, "mode": mode}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Expert migration cost vs hot-set turnover (Q3).")
    ap.add_argument("--fast", type=int, help="fast-tier CUDA index")
    ap.add_argument("--slow", type=int, help="slow-tier CUDA index")
    ap.add_argument("--capture", required=True, help="expert.jsonl from B3 (file or dir)")
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--intermediate", type=int, default=INTERMEDIATE)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--hot-k", type=int, default=0, help="hot-set size (0 = model top_k)")
    ap.add_argument("--turnover-threshold", type=float, default=0.5)
    ap.add_argument("--no-gpu", action="store_true",
                    help="skip PCIe measurement; use analytic bytes/Gen4 estimate")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from b2tel.analysis import build_windowed, turnover_period
    from b2tel.telemetry import read_jsonl

    ebytes = expert_param_bytes(args.hidden, args.intermediate, BF16_BYTES)

    # --- PCIe measurement (or analytic estimate) ---
    pcie: dict
    if args.no_gpu or args.fast is None or args.slow is None:
        # PCIe Gen4 x16 ≈ 25 GB/s achievable; analytic fallback when no node.
        est_gbps = 25.0
        pcie = {
            "ms_per_expert": (ebytes / 1e9) / est_gbps * 1e3,
            "GBps": est_gbps, "mode": "analytic-gen4-estimate",
        }
        print(f"[migrate_bench] analytic PCIe estimate ({est_gbps} GB/s)", file=sys.stderr)
    else:
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA")
            pcie = _measure_pcie(args.fast, args.slow, ebytes, args.iters, args.warmup)
        except Exception as e:  # noqa: BLE001
            print(f"[migrate_bench] PCIe measure failed ({e}); use --no-gpu", file=sys.stderr)
            return 2

    # --- hot-set turnover from the capture ---
    rows = read_jsonl(args.capture)
    w = build_windowed(rows)
    k = args.hot_k or w.top_k
    turn = turnover_period(w, k=k, threshold=args.turnover_threshold)

    mig_ms = pcie["ms_per_expert"]
    turn_s = turn["median_s"]
    if turn_s is None:  # never churned within capture → bounded below by the span
        turn_s = turn["capture_span_s"]
        turn_note = "censored (hot set stable for entire capture → turnover ≥ span)"
    else:
        turn_note = "measured median"
    headroom = (turn_s * 1000.0 / mig_ms) if mig_ms > 0 else None
    viable = headroom is not None and headroom >= 1.0

    out = {
        "expert_bytes": ebytes,
        "dtype": "bfloat16",
        "dims": {"hidden": args.hidden, "intermediate": args.intermediate},
        "pcie": pcie,
        "turnover": turn,
        "verdict": {
            "migration_ms": mig_ms,
            "turnover_period_s": turn_s,
            "turnover_note": turn_note,
            "headroom_x": headroom,
            "migrate_viable": viable,
            "policy": "migrate+replicate" if viable else "replication-only",
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[migrate_bench] expert={ebytes/1e6:.1f} MB  migrate={mig_ms:.3f} ms "
          f"({pcie['GBps']:.1f} GB/s, {pcie['mode']})")
    print(f"[migrate_bench] turnover={turn_s:.2f} s ({turn_note})  "
          f"headroom={headroom:.0f}x → {'MIGRATE VIABLE' if viable else 'REPLICATION-ONLY'}"
          if headroom else "[migrate_bench] headroom n/a")
    print(f"[migrate_bench] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
