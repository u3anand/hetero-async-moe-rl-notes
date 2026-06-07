#!/usr/bin/env python3
"""Tier micro-bench: one expert FFN's throughput on the fast vs slow GPU (step 4 / Q-tier).

Times a single OLMoE expert (a SwiGLU MLP: gate/up/down + SiLU) in BF16 at a range of batch
sizes on two CUDA devices — one **fast** (RTX 6000 Ada / L40S) and one **slow** (A6000,
Ampere) — and records tokens/s per tier. The fast:slow ratio is the size of the tier gap
the TierShift placement exploits (expect ≥ 2.4× BF16; wider with FP8-Ada, measured
separately). Output feeds the offline simulator's per-tier service rate.

BF16 only for OLMoE (the slow tier has no FP8 hardware; see HANDOFF "decisions locked").

Output ``tier_cost.json``::

    {
      "dtype": "bfloat16",
      "dims": {"hidden": 2048, "intermediate": 1024},
      "fast": {"index": <i>, "name": ...}, "slow": {"index": <j>, "name": ...},
      "per_batch": {
        "<batch>": {"fast_tok_s": ..., "slow_tok_s": ..., "ratio": ...}, ...
      }
    }

Usage:
  python -m b2tel.tier_bench --fast 6 --slow 4 --batch 1,8,32,128 \
      --out /u3/u3anand/b2/runs/tier_cost.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# OLMoE-1B-7B-0924 per-expert dims.
HIDDEN = 2048
INTERMEDIATE = 1024


def _expert(hidden: int, inter: int, dtype, device):
    import torch.nn as nn

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(hidden, inter, bias=False)
            self.up = nn.Linear(hidden, inter, bias=False)
            self.down = nn.Linear(inter, hidden, bias=False)
            self.act = nn.SiLU()

        def forward(self, x):
            return self.down(self.act(self.gate(x)) * self.up(x))

    return Expert().to(device=device, dtype=dtype).eval()


def _time_device(index: int, batches: list[int], hidden: int, inter: int,
                 iters: int, warmup: int) -> dict:
    import torch

    device = torch.device(f"cuda:{index}")
    dtype = torch.bfloat16
    name = torch.cuda.get_device_name(index)
    expert = _expert(hidden, inter, dtype, device)
    result = {"index": index, "name": name, "per_batch": {}}
    with torch.no_grad():
        for b in batches:
            x = torch.randn(b, hidden, device=device, dtype=dtype)
            for _ in range(warmup):
                expert(x)
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                expert(x)
            end.record()
            torch.cuda.synchronize(device)
            ms = start.elapsed_time(end) / iters
            tok_s = (b * 1000.0) / ms if ms > 0 else 0.0
            result["per_batch"][b] = {"ms_per_call": ms, "tok_s": tok_s}
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Expert-FFN tokens/s: fast tier vs slow tier.")
    ap.add_argument("--fast", type=int, required=True, help="fast-tier CUDA index (Ada/L40S)")
    ap.add_argument("--slow", type=int, required=True, help="slow-tier CUDA index (A6000)")
    ap.add_argument("--batch", default="1,8,32,128", help="comma list of batch sizes")
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--intermediate", type=int, default=INTERMEDIATE)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("[tier_bench] torch required (run on the cluster compute node)", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("[tier_bench] no CUDA — run inside a GPU job on watgpu308", file=sys.stderr)
        return 2

    batches = [int(x) for x in args.batch.split(",") if x.strip()]
    fast = _time_device(args.fast, batches, args.hidden, args.intermediate, args.iters, args.warmup)
    slow = _time_device(args.slow, batches, args.hidden, args.intermediate, args.iters, args.warmup)

    per_batch = {}
    for b in batches:
        ft = fast["per_batch"][b]["tok_s"]
        st = slow["per_batch"][b]["tok_s"]
        per_batch[str(b)] = {
            "fast_tok_s": ft, "slow_tok_s": st,
            "ratio": (ft / st) if st > 0 else None,
        }

    out = {
        "dtype": "bfloat16",
        "dims": {"hidden": args.hidden, "intermediate": args.intermediate},
        "fast": {"index": fast["index"], "name": fast["name"]},
        "slow": {"index": slow["index"], "name": slow["name"]},
        "per_batch": per_batch,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[tier_bench] {fast['name']} vs {slow['name']}")
    for b in batches:
        pb = per_batch[str(b)]
        r = pb["ratio"]
        print(f"  batch {b:>4}: fast {pb['fast_tok_s']:>10.0f} tok/s  "
              f"slow {pb['slow_tok_s']:>10.0f} tok/s  ratio {r:.2f}x" if r else f"  batch {b}: n/a")
    print(f"[tier_bench] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
