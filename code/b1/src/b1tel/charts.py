"""Q1–Q7 charts from the JSONL telemetry, rendered side-by-side across configs.

Reads ``runs/<config>/<ts>/telemetry/{episode,step,system}.jsonl`` for each config and
emits one figure per question into ``charts/``. Everything is derived here so the raw
JSONL stays the single source of truth.

Status: data-loading + Q1/Q5 implemented; Q2/Q3/Q4/Q6/Q7 stubbed (fill once a real run
produces data). Reporting discipline: p50/p95/p99, never bare means.

Usage:
    python -m b1tel.charts --runs homo-fast=runs/homo-fast/<ts> homo-slow=... hetero=... --out charts/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_stream(run_dir: str, stream: str) -> list[dict]:
    path = Path(run_dir) / "telemetry" / f"{stream}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def quantiles(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99))}


# ---- per-question renderers -------------------------------------------------

def q1_tool_breakdown(runs: dict[str, str], out: Path) -> None:
    """Stacked bar: seconds-by-tool-category, per config."""
    import matplotlib.pyplot as plt
    cats = ("inference", "file_op", "git", "mid_test", "build", "reward_eval")
    totals = {cfg: {c: 0.0 for c in cats} for cfg in runs}
    for cfg, d in runs.items():
        for ep in load_stream(d, "episode"):
            for tc in ep.get("tool_calls", []):
                totals[cfg][tc.get("category", "file_op")] += tc.get("dur_s", 0.0)
    fig, ax = plt.subplots()
    bottoms = {cfg: 0.0 for cfg in runs}
    for c in cats:
        ax.bar(list(runs), [totals[cfg][c] for cfg in runs],
               bottom=[bottoms[cfg] for cfg in runs], label=c)
        for cfg in runs:
            bottoms[cfg] += totals[cfg][c]
    ax.set_ylabel("seconds"); ax.set_title("Q1 per-episode wallclock by category"); ax.legend()
    fig.savefig(out / "q1_tool_breakdown.png", dpi=150, bbox_inches="tight")


def q5_episode_cdf(runs: dict[str, str], out: Path) -> None:
    """CDF of per-episode wallclock with p50/p95/p99 marked."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for cfg, d in runs.items():
        walls = [ep["t_end"] - ep["t_start"] for ep in load_stream(d, "episode")
                 if ep.get("t_end") and ep.get("t_start")]
        if not walls:
            continue
        xs = np.sort(walls)
        ax.plot(xs, np.linspace(0, 1, len(xs)), label=f"{cfg} (p95={quantiles(walls)['p95']:.0f}s)")
    ax.set_xlabel("episode wallclock (s)"); ax.set_ylabel("CDF")
    ax.set_title("Q5 episode wallclock CDF"); ax.legend()
    fig.savefig(out / "q5_episode_cdf.png", dpi=150, bbox_inches="tight")


def q2_gpu_starvation(runs: dict[str, str], out: Path) -> None:
    """TODO: dual-axis GPU busy% vs sandbox queue depth → M4 trigger."""


def q3_ci_ct(runs: dict[str, str], out: Path) -> None:
    """TODO: C_I (split C_infer/C_tool) vs C_T per config."""


def q4_broadcast_split(runs: dict[str, str], out: Path) -> None:
    """TODO: router vs expert broadcast bytes/time (smoke-test under LoRA)."""


def q6_router_kl(runs: dict[str, str], out: Path) -> None:
    """TODO: cross-replica router-KL vs tier-speed gap."""


def q7_mfu_hbm(runs: dict[str, str], out: Path) -> None:
    """TODO: train MFU + rollout HBM-bw utilization per tier."""


RENDERERS = [q1_tool_breakdown, q2_gpu_starvation, q3_ci_ct, q4_broadcast_split,
             q5_episode_cdf, q6_router_kl, q7_mfu_hbm]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="config=run_dir pairs")
    ap.add_argument("--out", default="charts")
    args = ap.parse_args()
    runs = dict(p.split("=", 1) for p in args.runs)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for render in RENDERERS:
        render(runs, out)
    print(f"charts → {out}")


if __name__ == "__main__":
    main()
