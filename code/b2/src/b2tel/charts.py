#!/usr/bin/env python3
"""Render the Q1–Q7 figures from the B2 artifacts (Initial Plan 2 → step 6).

Reads the captured ``expert.jsonl`` (+ optional ``segments.jsonl``), the simulator's
``sim.json``, and the two micro-bench JSONs, and writes PNGs:

  q1_imbalance_over_time.png   non-stationarity: imbalance ratio over wall-clock (headline)
  q2_slo_and_cost.png          p99/SLO-attainment + $/token per config (the objective)
  q3_migration_vs_turnover.png migration-ms vs hot-set-turnover-period (the kill-chart)
  q4_config_headroom.png       per-config p99 + the three gaps (GEM gap; make-or-break)
  q5_reactive_hit_rate.png     does last window's hot set capture this window's (cache precond)
  q6_domain_heatmap.png        hot-set shift by domain (chat/code/math/mixed)
  q_taxonomy.png               consistent vs temporal experts + temporal-mass fraction

Headless (Agg backend). Each chart is best-effort: a missing input is logged and skipped, so
partial runs still produce whatever they can.

Usage:
  python -m b2tel.charts --capture .../expert.jsonl --segments .../segments.jsonl \
      --sim .../sim.json --tier-cost .../tier_cost.json --migration .../migration_cost.json \
      --out code/b2/charts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from b2tel.analysis import (  # noqa: E402
    build_windowed, classify_experts, hot_set, imbalance_series, reactive_hit_rate,
)
from b2tel.telemetry import assign_segments, read_jsonl  # noqa: E402

SEG_ORDER = ["chat", "code", "math", "mixed"]


def _save(fig, out: Path, name: str):
    path = out / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[charts] wrote {path}")


def chart_q1(w, segs, out: Path):
    times, imb = imbalance_series(w)
    t0 = times[0] if times else 0
    rel = [t - t0 for t in times]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rel, imb, lw=1.6, color="#1f77b4")
    ax.set_xlabel("wall-clock (s)")
    ax.set_ylabel("imbalance ratio (max/mean load)")
    ax.set_title("Q1 — expert-load non-stationarity over time")
    _shade_segments(ax, segs, t0)
    ax.axhline(1.0, ls=":", c="gray", lw=0.8)
    _save(fig, out, "q1_imbalance_over_time.png")


def _shade_segments(ax, segs, t0):
    if not segs:
        return
    colors = {"chat": "#cfe8ff", "code": "#d5f0d5", "math": "#ffe0cc", "mixed": "#eee0ff"}
    for s in segs:
        ax.axvspan(s["t_start"] - t0, s["t_end"] - t0,
                   color=colors.get(s["segment_label"], "#f0f0f0"), alpha=0.5, lw=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors.get(k, "#f0f0f0")) for k in SEG_ORDER]
    ax.legend(handles, SEG_ORDER, loc="upper right", fontsize=8, ncol=4)


def chart_q2(sim, out: Path):
    cfgs = list(sim["configs"].keys())
    p99 = [sim["configs"][c]["p99_ms"] for c in cfgs]
    cost = [sim["configs"][c]["usd_per_1k_tokens"] or 0 for c in cfgs]
    slo = sim["params"]["slo_p99_ms"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(cfgs))
    a1.bar(x, p99, color="#4c72b0")
    a1.axhline(slo, ls="--", c="red", label=f"SLO p99 = {slo:.0f} ms")
    a1.set_xticks(x); a1.set_xticklabels(cfgs, rotation=40, ha="right", fontsize=8)
    a1.set_ylabel("p99 window latency (ms)"); a1.set_title("Q2 — p99 vs SLO under drift")
    a1.legend(fontsize=8)
    a2.bar(x, cost, color="#55a868")
    ref = sim.get("all_fast_reference", {}).get("usd_per_1k_tokens")
    if ref:
        a2.axhline(ref, ls="--", c="purple", label=f"all-fast ref = {ref:.4f}")
        a2.legend(fontsize=8)
    a2.set_xticks(x); a2.set_xticklabels(cfgs, rotation=40, ha="right", fontsize=8)
    a2.set_ylabel("$ / 1k tokens"); a2.set_title("Q2 — cost per token (objective)")
    _save(fig, out, "q2_slo_and_cost.png")


def chart_q3(mig, out: Path):
    v = mig["verdict"]
    mig_ms = v["migration_ms"]
    turn_ms = (v["turnover_period_s"] or 0) * 1000
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(["migration\n(1 expert, PCIe)", "hot-set turnover\nperiod"],
                  [mig_ms, turn_ms], color=["#c44e52", "#4c72b0"])
    ax.set_ylabel("milliseconds (log)")
    ax.set_yscale("log")
    ax.set_title(f"Q3 — kill-test: {'MIGRATE VIABLE' if v['migrate_viable'] else 'REPLICATION-ONLY'}"
                 f"  ({v.get('headroom_x') or 0:.0f}x headroom)")
    for b, val in zip(bars, [mig_ms, turn_ms]):
        ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.1f} ms",
                ha="center", va="bottom", fontsize=9)
    _save(fig, out, "q3_migration_vs_turnover.png")


def chart_q4(sim, out: Path):
    cfgs = list(sim["configs"].keys())
    p99 = [sim["configs"][c]["p99_ms"] for c in cfgs]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(cfgs))
    colors = ["#bbb"] * len(cfgs)
    for i, c in enumerate(cfgs):
        if c == "GEM-style": colors[i] = "#dd8452"
        if c == "reactive-LRU-port": colors[i] = "#4c72b0"
        if c == "reactive+split": colors[i] = "#55a868"
        if c == "oracle-dynamic": colors[i] = "#8172b3"
    ax.bar(x, p99, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("p99 latency (ms)")
    g = sim["gaps"]
    ax.set_title(f"Q4 — gaps: GEM→split {g['GEM_gap']['p99_improvement_pct']:.0f}%  |  "
                 f"LRU-port→split {g['make_or_break']['p99_improvement_pct']:.0f}%  |  "
                 f"static-avg→oracle {g['static_avg_to_oracle']['p99_improvement_pct']:.0f}%")
    _save(fig, out, "q4_config_headroom.png")


def chart_q5(w, out: Path):
    hit = reactive_hit_rate(w, k=w.top_k)
    if not hit["per_window"]:
        print("[charts] q5: too few windows", file=sys.stderr); return
    t0 = hit["times"][0]
    rel = [t - t0 for t in hit["times"]]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rel, hit["per_window"], lw=1.4, color="#55a868")
    ax.axhline(hit["mean_hit_rate"], ls="--", c="gray",
               label=f"mean = {hit['mean_hit_rate']:.2f}")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("wall-clock (s)"); ax.set_ylabel("reactive cache hit-rate")
    ax.set_title("Q5 — does last window's hot set capture this window's? (no forecast)")
    ax.legend(fontsize=8)
    _save(fig, out, "q5_reactive_hit_rate.png")


def chart_q6(w, segs, out: Path):
    if not segs:
        print("[charts] q6: no segments.jsonl, skipping domain heatmap", file=sys.stderr)
        return
    # mean per-expert load (summed over layers) per domain → heatmap [domain x expert]
    labels = [s for s in SEG_ORDER if any(sg["segment_label"] == s for sg in segs)]
    mat = np.zeros((len(labels), w.n_experts))
    seg_by_label = {}
    for sg in segs:
        seg_by_label.setdefault(sg["segment_label"], []).append((sg["t_start"], sg["t_end"]))
    for li, lab in enumerate(labels):
        spans = seg_by_label[lab]
        for i, wid in enumerate(w.window_ids):
            t = w.times[i]
            if any(a <= t <= b for a, b in spans):
                mat[li] += w.total_load(wid)
    # normalize each row to a distribution
    rs = mat.sum(axis=1, keepdims=True)
    mat = np.divide(mat, rs, out=np.zeros_like(mat), where=rs > 0)
    fig, ax = plt.subplots(figsize=(11, 3 + 0.3 * len(labels)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("expert id"); ax.set_title("Q6 — hot-set shift by domain (normalized load)")
    fig.colorbar(im, ax=ax, fraction=0.025)
    _save(fig, out, "q6_domain_heatmap.png")


def chart_taxonomy(w, sim, out: Path):
    tax = classify_experts(w, k=w.top_k)
    frac = tax["temporal_mass_fraction"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.bar(["consistent", "temporal"], [tax["n_consistent"], tax["n_temporal"]],
           color=["#4c72b0", "#dd8452"])
    a1.set_ylabel("# experts (hot in ≥1 window)")
    a1.set_title("GEM taxonomy — consistent vs temporal experts")
    a2.bar(["consistent", "temporal"], [tax["consistent_mass"], tax["temporal_mass"]],
           color=["#4c72b0", "#dd8452"])
    a2.set_ylabel("hot-token mass")
    a2.set_title(f"temporal-mass fraction = {frac:.2f}\n(online-over-static ceiling)"
                 if frac is not None else "temporal-mass fraction = n/a")
    _save(fig, out, "q_taxonomy.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render B2 Q1–Q7 charts.")
    ap.add_argument("--capture", help="expert.jsonl (file or dir)")
    ap.add_argument("--segments", default=None)
    ap.add_argument("--sim", help="sim.json")
    ap.add_argument("--tier-cost", default=None)
    ap.add_argument("--migration", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    w = segs = None
    if args.capture and Path(args.capture).exists():
        rows = read_jsonl(args.capture)
        segs = read_jsonl(args.segments) if args.segments and Path(args.segments).exists() else None
        if segs:
            assign_segments(rows, segs)
        w = build_windowed(rows)

    sim = json.loads(Path(args.sim).read_text()) if args.sim and Path(args.sim).exists() else None
    mig = json.loads(Path(args.migration).read_text()) if args.migration and Path(args.migration).exists() else None

    if w is not None:
        chart_q1(w, segs, out)
        chart_q5(w, out)
        chart_q6(w, segs, out)
    if sim is not None:
        chart_q2(sim, out)
        chart_q4(sim, out)
        if w is not None:
            chart_taxonomy(w, sim, out)
    if mig is not None:
        chart_q3(mig, out)
    print(f"[charts] done → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
