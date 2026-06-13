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
    build_windowed, classify_experts, hot_set, hot_streak_cdf, imbalance_series,
    microbatch_vs_window, per_gpu_imbalance_series, reactive_hit_rate, topn_mass_skew,
)
from b2tel.telemetry import assign_segments, read_jsonl  # noqa: E402

PHASES = ["prefill", "decode"]

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


def chart_hot_streak(w, out: Path):
    """Q8 — CDF of how long an expert stays hot once hot (Khuzaima's ask)."""
    hs = hot_streak_cdf(w)
    if not hs["lengths_windows"]:
        print("[charts] hot-streak: no streaks, skipping")
        return
    dt = hs["window_dt_s"]
    xs = [n * dt for n in hs["lengths_windows"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(xs, hs["cdf"], where="post", color="#1f77b4", lw=1.8)
    if hs["median_s"]:
        ax.axvline(hs["median_s"], ls="--", c="#d62728", lw=1,
                   label=f"median {hs['median_s']:.1f}s ({hs['median_windows']:.0f} win)")
        ax.axvline(dt, ls=":", c="gray", lw=0.8, label=f"1 window = {dt:.1f}s")
        ax.legend(fontsize=8)
    ax.set_xlabel("hot-streak length (s)")
    ax.set_ylabel("CDF")
    ax.set_title("Q8 — once hot, how long does an expert stay hot?")
    _save(fig, out, "q8_hot_streak_cdf.png")


def chart_phase_imbalance(rows, n_gpus: int, out: Path):
    """Prefill vs decode: per-GPU token-load and activated-expert imbalance (the verdict
    metric). Activated-expert is the decode-cost metric that can stay skewed when token-load
    looks flat."""
    means = {}
    for ph in PHASES:
        try:
            wp = build_windowed(rows, phase=ph)
        except ValueError:
            continue
        _, tok, act = per_gpu_imbalance_series(wp, n_gpus)
        means[ph] = (float(tok.mean()), float(act.mean()))
    if not means:
        print("[charts] phase imbalance: no prefill/decode rows, skipping")
        return
    phs = list(means)
    x = np.arange(len(phs))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, [means[p][0] for p in phs], 0.4, label="token-load", color="#1f77b4")
    ax.bar(x + 0.2, [means[p][1] for p in phs], 0.4, label="activated-expert", color="#ff7f0e")
    ax.axhline(1.0, ls=":", c="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(phs)
    ax.set_ylabel(f"per-GPU imbalance (max/mean, D={n_gpus})")
    ax.set_title("Prefill vs decode imbalance (token-load vs activated-expert)")
    ax.legend(fontsize=8)
    _save(fig, out, "q_phase_imbalance.png")


def chart_topn_skew(w, out: Path):
    """Placement-free generality verdict: top-N mass vs the matched balls-in-bins null."""
    sk = topn_mass_skew(w, n=w.top_k, n_trials=20)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["observed", "uniform null"],
           [sk["observed_topn_mass"], sk["null_topn_mass"]],
           color=["#2ca02c", "#999999"])
    r = sk["ratio"]
    ax.set_ylabel(f"top-{sk['n']} mass (of {sk['n_experts']} experts)")
    ax.set_title(f"Routing skew vs matched null — ratio {r:.2f}× "
                 f"({'≥2× skewed' if r and r >= 2 else 'near-uniform'})")
    _save(fig, out, "q_topn_mass_skew.png")


def chart_microbatch(mb_rows, rows, n_gpus: int, out: Path):
    """M1 — per-micro-batch vs windowed imbalance (is a seconds-granularity cache enough?)."""
    labels, win, mb = [], [], []
    for ph in PHASES:
        try:
            wp = build_windowed(rows, phase=ph)
        except ValueError:
            continue
        r = microbatch_vs_window(mb_rows, wp, n_gpus, phase=ph)
        if r["microbatch_act_imbalance_mean"] is None:
            continue
        labels.append(ph)
        win.append(r["window_act_imbalance"])
        mb.append(r["microbatch_act_imbalance_mean"])
    if not labels:
        print("[charts] microbatch: no data, skipping")
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, win, 0.4, label="window (what cache sees)", color="#1f77b4")
    ax.bar(x + 0.2, mb, 0.4, label="micro-batch (what all-to-all suffers)", color="#d62728")
    ax.axhline(1.0, ls=":", c="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"activated-expert imbalance (D={n_gpus})")
    ax.set_title("Micro-batch vs window imbalance (cache timescale / Lina)")
    ax.legend(fontsize=8)
    _save(fig, out, "q_microbatch_vs_window.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render B2 Q1–Q7 charts.")
    ap.add_argument("--capture", help="expert.jsonl (file or dir)")
    ap.add_argument("--segments", default=None)
    ap.add_argument("--sim", help="sim.json")
    ap.add_argument("--tier-cost", default=None)
    ap.add_argument("--migration", default=None)
    ap.add_argument("--sim-gpus", type=int, default=8,
                    help="simulated EP GPUs for per-GPU imbalance (match VLLM_B2_SIM_GPUS)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    w = segs = rows = None
    mb_rows: list[dict] = []
    if args.capture and Path(args.capture).exists():
        rows = read_jsonl(args.capture)
        segs = read_jsonl(args.segments) if args.segments and Path(args.segments).exists() else None
        if segs:
            assign_segments(rows, segs)
        w = build_windowed(rows)  # filters out the mb summary rows internally
        # micro-batch summaries live in sibling expert_mb.*.jsonl
        cap = Path(args.capture)
        cap_dir = cap if cap.is_dir() else cap.parent
        for f in sorted(cap_dir.glob("expert_mb.*.jsonl")):
            mb_rows += read_jsonl(f)

    sim = json.loads(Path(args.sim).read_text()) if args.sim and Path(args.sim).exists() else None
    mig = json.loads(Path(args.migration).read_text()) if args.migration and Path(args.migration).exists() else None

    if w is not None:
        chart_q1(w, segs, out)
        chart_q5(w, out)
        chart_q6(w, segs, out)
        chart_hot_streak(w, out)
        chart_topn_skew(w, out)
        chart_phase_imbalance(rows, args.sim_gpus, out)
        if mb_rows:
            chart_microbatch(mb_rows, rows, args.sim_gpus, out)
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
