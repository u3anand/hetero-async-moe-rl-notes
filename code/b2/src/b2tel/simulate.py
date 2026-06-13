#!/usr/bin/env python3
"""Offline placement simulator — the B6 headline (Initial Plan 2 → step 6, Q2/Q4).

Joins the captured per-window expert load (``expert.jsonl``) with the measured tier gap
(``tier_cost.json``), migration cost (``migration_cost.json``), and fleet pricing
(``fleet_cost.json``), and scores eight expert→tier placement policies on the **objective**:
**p99 window latency / SLO-attainment + $/token**. The model + trace are held constant; only
the expert→tier assignment changes (controlled comparison).

Model (documented assumptions — this is a characterization sim, not a cycle model):
- The fleet is ``n_fast`` fast GPUs + ``n_slow`` slow GPUs (from fleet_cost.json). A fast GPU
  serves expert-FFN tokens at ``fast_tok_s``, a slow GPU at ``slow_tok_s`` (from tier_cost at
  a chosen batch). The fast tier can hold ``cap_fast`` resident experts (cache budget); the
  rest live on the slow tier. Every expert is served somewhere every window.
- Per window we place experts on tiers, bin-pack each tier's resident experts across its GPUs
  by load (LPT), and the tier's service time is ``max_gpu_load / tok_s``. The window's
  **latency = max(fast_time, slow_time)** (the straggler). Splitting the hottest expert across
  fast GPUs lowers the fast-tier max (that's reactive+split's lever).
- Reactive policies that change the fast set pay a **migration penalty** = (#experts entering
  fast) × ``migration_ms``, added to that window's fast_time — unless migration is not viable
  (then the migrate-configs are penalised and the honest fallback is replication-only).
- **$/token** = fleet $/hr × (sum of window latencies, as hours) ÷ total tokens served. A
  placement that finishes windows faster costs less. The **all-fast reference** prices a fleet
  of only fast GPUs sized to hold every expert.

Configs (Initial Plan 2): static-balanced, static-avg-opt, GEM-style, HarMoEny-style,
CRAFT-style, reactive-LRU-port, reactive+split, oracle-dynamic. Plus the all-fast reference.

Reports the **three gaps**: static-avg→oracle, **GEM→reactive+split (the GEM gap)**, and
reactive-LRU-port→reactive+split (make-or-break), plus the temporal-mass fraction.

Usage:
  python -m b2tel.simulate --capture .../expert.jsonl --segments .../segments.jsonl \
      --tier-cost .../tier_cost.json --migration .../migration_cost.json \
      --fleet-cost code/b2/fleet_cost.json --batch 8 --slo-p99-ms 0 \
      --out /u3/u3anand/b2/runs/sim.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from b2tel.analysis import (
    Windowed, build_windowed, classify_experts, hot_set, imbalance_series,
    reactive_hit_rate, turnover_period,
)
from b2tel.telemetry import assign_segments, read_jsonl

CONFIGS = [
    "static-balanced", "static-avg-opt", "GEM-style", "HarMoEny-style",
    "CRAFT-style", "reactive-LRU-port", "reactive+split", "oracle-dynamic",
]


def _expert_keys(w: Windowed) -> list[tuple[int, int]]:
    """All (layer, expert) physical experts (global cache namespace)."""
    return [(layer, e) for layer in w.layers for e in range(w.n_experts)]


def _flat_load(w: Windowed, wid: int) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for layer in w.layers:
        loads = w.layer_load(wid, layer)
        for e in range(w.n_experts):
            v = float(loads[e])
            if v > 0:
                out[(layer, e)] = v
    return out


def _fit_two_term(tier: dict, which: str, batch_key: str) -> tuple[float, float]:
    """Return (load_weights_s, per_token_s) for tier ``which`` ('fast'/'slow').

    Two-point fit of ``{which}_ms_per_call`` over batch size → cost(n) = load_weights +
    n·per_token. The fixed weight-load term is what makes the decode regime honest: each
    expert (and each split replica) pays a full weight load every step it is active,
    regardless of its token count, so replicate-and-split is no longer free. Falls back to the
    legacy pure-throughput model (no fixed term) when ``ms_per_call`` is absent."""
    pb = tier["per_batch"]
    ms_key = f"{which}_ms_per_call"
    pts = [(int(b), float(d[ms_key])) for b, d in pb.items() if d.get(ms_key) is not None]
    if len(pts) >= 2:
        pts.sort()
        (nlo, mlo), (nhi, mhi) = pts[0], pts[-1]
        if nhi > nlo:
            per_token_ms = max(1e-9, (mhi - mlo) / (nhi - nlo))
            load_weights_ms = max(0.0, mlo - per_token_ms * nlo)
            return load_weights_ms / 1000.0, per_token_ms / 1000.0
    # legacy / single-batch: pure throughput at the chosen batch, no fixed term
    d = pb.get(batch_key) or next(iter(pb.values()))
    tok_s = float(d[f"{which}_tok_s"])
    return 0.0, (1.0 / tok_s if tok_s > 0 else float("inf"))


def _lpt_makespan(
    loads: list[float], n_bins: int, load_weights_s: float, per_token_s: float
) -> float:
    """Longest-processing-time bin-packing; return max-bin **time** (seconds).

    Each active expert contributes ``load_weights_s`` (fixed weight load, paid once per step)
    plus ``load·per_token_s`` (marginal). Packing by per-item time means more distinct experts
    on a GPU costs more even at equal token load — the decode/weight-load reality."""
    if n_bins <= 0 or per_token_s == float("inf"):
        return float("inf")
    bins = np.zeros(n_bins)
    items = sorted((load_weights_s + L * per_token_s for L in loads), reverse=True)
    for t in items:
        bins[int(np.argmin(bins))] += t
    return float(bins.max())


def _tier_time(
    fast_loads: list[float], slow_loads: list[float],
    n_fast: int, n_slow: int, fast_cost: tuple[float, float], slow_cost: tuple[float, float],
    split_hottest: bool,
) -> tuple[float, float]:
    """Per-tier service time. ``split_hottest`` divides the single largest fast expert
    evenly across the fast GPUs before packing the rest (the reactive+split lever) — each
    replica is now a separate item, so it pays its own fixed weight load (decode break-even)."""
    fl = list(fast_loads)
    if split_hottest and fl and n_fast > 1:
        fl.sort(reverse=True)
        hot = fl.pop(0)
        share = hot / n_fast
        fl.extend([share] * n_fast)  # the split replicas (each pays load_weights)
    fast_t = _lpt_makespan(fl, n_fast, *fast_cost) if fl else 0.0
    slow_t = _lpt_makespan(slow_loads, n_slow, *slow_cost) if slow_loads else 0.0
    return fast_t, slow_t


def _demand_set(w: Windowed, wid: int, k: int) -> set[tuple[int, int]]:
    """The genuine hot set for a window: per-layer top-k experts (never the cold tail). This
    is what a reactive cache *demands* be resident — cold experts are never demanded, so the
    cache doesn't thrash on load ties in the long tail."""
    s: set[tuple[int, int]] = set()
    for layer in w.layers:
        for e in hot_set(w.layer_load(wid, layer), k):
            s.add((layer, e))
    return s


def _static_fast_set(
    config: str, w: Windowed, cap_fast: int, global_mean: dict
) -> tuple[set[tuple[int, int]], bool]:
    """Fixed fast set + split flag for the static configs (computed once)."""
    if config == "static-balanced":
        keys = _expert_keys(w)
        step = max(1, len(keys) // max(1, cap_fast))
        return set(keys[::step][:cap_fast]), False
    # static-avg-opt / GEM-style / CRAFT-style: hot→fast from the whole-trace average.
    active = sorted((k for k, v in global_mean.items() if v > 0),
                    key=lambda k: global_mean[k], reverse=True)
    # GEM is speed-aware (split); CRAFT replicates the hottest; avg-opt is plain placement.
    return set(active[:cap_fast]), config in ("GEM-style", "CRAFT-style")


def _sticky_admit(
    resident: set, last_seen: dict, demand: set, cap: int, t: int
) -> int:
    """Update a sticky LRU cache toward ``demand`` within ``cap``. Residents that are still
    demanded refresh their recency; demanded experts not resident are admitted, evicting the
    least-recently-demanded non-demanded resident when full. Returns # admitted (= weights
    migrated onto the fast tier this window)."""
    for e in demand:
        if e in resident:
            last_seen[e] = t
    admitted = 0
    for e in demand:
        if e in resident:
            continue
        if len(resident) >= cap:
            evictable = [r for r in resident if r not in demand]
            if not evictable:
                break  # cache saturated with currently-demanded experts
            victim = min(evictable, key=lambda r: last_seen.get(r, -1))
            resident.discard(victim)
            last_seen.pop(victim, None)
        resident.add(e)
        last_seen[e] = t
        admitted += 1
    return admitted


def simulate_config(
    config: str, w: Windowed, fleet: dict,
    fast_cost: tuple[float, float], slow_cost: tuple[float, float],
    cap_fast: int, migration_ms: float, migrate_viable: bool,
    global_mean: dict[tuple[int, int], float], hot_k: int,
) -> dict:
    n_fast = int(fleet["fast"]["count"])
    n_slow = int(fleet["slow"]["count"])
    lat = []  # per-window latency (s)
    migrations = 0
    # only the *deployable* reactive caches move expert weights and thus pay migration cost.
    # oracle-dynamic is the idealized placement ceiling (no migration penalty); HarMoEny
    # redistributes *tokens*, not weights, so it doesn't migrate experts either.
    deployable_reactive = config in ("reactive-LRU-port", "reactive+split")
    is_static = config in ("static-balanced", "static-avg-opt", "GEM-style", "CRAFT-style")
    static_fast, static_split = (
        _static_fast_set(config, w, cap_fast, global_mean) if is_static else (set(), False)
    )
    resident: set[tuple[int, int]] = set()
    last_seen: dict[tuple[int, int], int] = {}

    for i, wid in enumerate(w.window_ids):
        cur = _flat_load(w, wid)
        admitted = 0
        # all dynamic configs FILL the fast tier with their hottest cap_fast experts
        # (apples-to-apples with static, which also fills cap_fast)
        def _topn(loadmap, n):
            return set(sorted((k for k, v in loadmap.items() if v > 0),
                              key=lambda k: loadmap[k], reverse=True)[:n])
        if is_static:
            fast_set, split = static_fast, static_split
        elif config == "oracle-dynamic":
            fast_set, split = _topn(cur, cap_fast), True                  # current window, ceiling
        elif config == "HarMoEny-style":
            fast_set, split = _topn(cur, cap_fast), False                 # token-balance, no tier term
        else:  # reactive caches react to the *previous* window (no forecast)
            sig = cur if i == 0 else _flat_load(w, w.window_ids[i - 1])
            demand = _topn(sig, cap_fast)
            admitted = _sticky_admit(resident, last_seen, demand, cap_fast, i)
            fast_set, split = set(resident), (config == "reactive+split")

        fast_loads = [cur[k] for k in fast_set if k in cur]
        slow_loads = [v for k, v in cur.items() if k not in fast_set]
        fast_t, slow_t = _tier_time(
            fast_loads, slow_loads, n_fast, n_slow, fast_cost, slow_cost, split
        )
        if deployable_reactive:
            migrations += admitted
            pen_ms = migration_ms if migrate_viable else migration_ms * 4.0  # stall if not viable
            fast_t += admitted * pen_ms / 1000.0
        lat.append(max(fast_t, slow_t))

    lat_a = np.array(lat)
    total_tokens = sum(float(w.total_load(wid).sum()) for wid in w.window_ids)
    wall_s = float(lat_a.sum())
    fleet_hr = n_fast * fleet["fast"]["usd_per_hr"] + n_slow * fleet["slow"]["usd_per_hr"]
    usd_per_1k = (fleet_hr * (wall_s / 3600.0) / total_tokens * 1000.0) if total_tokens else None
    return {
        "config": config,
        "p50_ms": float(np.percentile(lat_a, 50) * 1000),
        "p95_ms": float(np.percentile(lat_a, 95) * 1000),
        "p99_ms": float(np.percentile(lat_a, 99) * 1000),
        "mean_ms": float(lat_a.mean() * 1000),
        "max_ms": float(lat_a.max() * 1000),
        "usd_per_1k_tokens": usd_per_1k,
        "usd_per_1m_tokens": usd_per_1k * 1000 if usd_per_1k is not None else None,
        "migrations_total": migrations,
        "wall_s": wall_s,
        "lat_ms_series": (lat_a * 1000).tolist(),
    }


def all_fast_reference(
    w: Windowed, fleet: dict, fast_cost: tuple[float, float], cap_fast_per_gpu: int
) -> dict:
    """All experts on a fast-only fleet sized to hold them (the cost-at-SLO reference)."""
    n_experts_total = len(w.layers) * w.n_experts
    n_fast = max(1, -(-n_experts_total // max(1, cap_fast_per_gpu)))  # ceil
    lat = []
    for wid in w.window_ids:
        cur = _flat_load(w, wid)
        lat.append(_lpt_makespan(list(cur.values()), n_fast, *fast_cost))
    lat_a = np.array(lat)
    total_tokens = sum(float(w.total_load(wid).sum()) for wid in w.window_ids)
    wall_s = float(lat_a.sum())
    fleet_hr = n_fast * fleet["fast"]["usd_per_hr"]
    usd_per_1k = (fleet_hr * (wall_s / 3600.0) / total_tokens * 1000.0) if total_tokens else None
    return {
        "config": "all-fast-reference", "n_fast": n_fast,
        "p99_ms": float(np.percentile(lat_a, 99) * 1000),
        "mean_ms": float(lat_a.mean() * 1000),
        "usd_per_1k_tokens": usd_per_1k, "wall_s": wall_s,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B2 offline placement simulator (Q2/Q4).")
    ap.add_argument("--capture", required=True, help="expert.jsonl (file or capture dir)")
    ap.add_argument("--segments", default=None, help="segments.jsonl (label windows by domain)")
    ap.add_argument("--tier-cost", required=True)
    ap.add_argument("--migration", required=True)
    ap.add_argument("--fleet-cost", required=True)
    ap.add_argument("--batch", type=int, default=8, help="batch size key into tier_cost")
    ap.add_argument("--cap-fast-frac", type=float, default=0.25,
                    help="fraction of all experts the fast tier can hold resident")
    ap.add_argument("--cap-fast-per-gpu", type=int, default=64,
                    help="experts a single fast GPU holds (for the all-fast reference)")
    ap.add_argument("--slo-p99-ms", type=float, default=0.0,
                    help="p99 SLO target; 0 = auto (1.5x the oracle p99)")
    ap.add_argument("--consistent-frac", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows = read_jsonl(args.capture)
    if args.segments and Path(args.segments).exists():
        assign_segments(rows, read_jsonl(args.segments))
    w = build_windowed(rows)

    tier = json.loads(Path(args.tier_cost).read_text())
    mig = json.loads(Path(args.migration).read_text())
    fleet = json.loads(Path(args.fleet_cost).read_text())

    bkey = str(args.batch)
    pb = tier["per_batch"].get(bkey) or next(iter(tier["per_batch"].values()))
    fast_tok_s = float(pb["fast_tok_s"])
    slow_tok_s = float(pb["slow_tok_s"])
    # two-term cost: (load_weights_s, per_token_s) per tier — fixed weight load + marginal.
    fast_cost = _fit_two_term(tier, "fast", bkey)
    slow_cost = _fit_two_term(tier, "slow", bkey)
    migration_ms = float(mig["verdict"]["migration_ms"])
    migrate_viable = bool(mig["verdict"]["migrate_viable"])

    n_experts_total = len(w.layers) * w.n_experts
    cap_fast = max(1, int(round(args.cap_fast_frac * n_experts_total)))

    # whole-trace average load per expert (for the static mappings)
    global_mean: dict[tuple[int, int], float] = {}
    for wid in w.window_ids:
        for k, v in _flat_load(w, wid).items():
            global_mean[k] = global_mean.get(k, 0.0) + v

    results = {
        c: simulate_config(c, w, fleet, fast_cost, slow_cost, cap_fast,
                           migration_ms, migrate_viable, global_mean, w.top_k)
        for c in CONFIGS
    }
    ref = all_fast_reference(w, fleet, fast_cost, args.cap_fast_per_gpu)

    # derived: taxonomy, locality, the three gaps
    taxonomy = classify_experts(w, k=w.top_k, consistent_frac=args.consistent_frac)
    hit = reactive_hit_rate(w, k=w.top_k)
    turn = turnover_period(w, k=w.top_k)
    times, imb = imbalance_series(w)

    slo = args.slo_p99_ms or 1.5 * results["oracle-dynamic"]["p99_ms"]
    for c, r in results.items():
        series = np.array(r["lat_ms_series"])
        r["slo_attainment"] = float((series <= slo).mean())

    def gap(a: str, b: str) -> dict:
        ra, rb = results[a], results[b]
        return {
            "from": a, "to": b,
            "p99_ms_from": ra["p99_ms"], "p99_ms_to": rb["p99_ms"],
            "p99_improvement_pct": (ra["p99_ms"] - rb["p99_ms"]) / ra["p99_ms"] * 100
            if ra["p99_ms"] else None,
            "usd_per_1k_from": ra["usd_per_1k_tokens"], "usd_per_1k_to": rb["usd_per_1k_tokens"],
        }

    gaps = {
        "static_avg_to_oracle": gap("static-avg-opt", "oracle-dynamic"),
        "GEM_gap": gap("GEM-style", "reactive+split"),
        "make_or_break": gap("reactive-LRU-port", "reactive+split"),
    }

    out = {
        "params": {
            "batch": args.batch, "fast_tok_s": fast_tok_s, "slow_tok_s": slow_tok_s,
            "tier_ratio": fast_tok_s / slow_tok_s if slow_tok_s else None,
            "fast_cost_s": {"load_weights": fast_cost[0], "per_token": fast_cost[1]},
            "slow_cost_s": {"load_weights": slow_cost[0], "per_token": slow_cost[1]},
            "cap_fast": cap_fast, "n_experts_total": n_experts_total,
            "migration_ms": migration_ms, "migrate_viable": migrate_viable,
            "slo_p99_ms": slo, "n_windows": len(w.window_ids),
        },
        "configs": results,
        "all_fast_reference": ref,
        "gaps": gaps,
        "temporal_mass_fraction": taxonomy["temporal_mass_fraction"],
        "taxonomy_summary": {k: taxonomy[k] for k in (
            "n_consistent", "n_temporal", "consistent_mass", "temporal_mass",
            "temporal_mass_fraction")},
        "reactive_hit_rate": hit["mean_hit_rate"],
        "turnover": turn,
        "imbalance": {"times": times, "ratio": imb.tolist(),
                      "min": float(imb.min()), "max": float(imb.max()),
                      "swing": float(imb.max() - imb.min())},
    }
    # keep the big per-window series out of the headline but available
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"[simulate] {len(w.window_ids)} windows, tier ratio "
          f"{out['params']['tier_ratio']:.2f}x, SLO p99 ≤ {slo:.1f} ms")
    print(f"{'config':>20}  {'p99_ms':>9}  {'$/1M tok':>9}  {'SLO%':>6}  {'migr':>6}")
    for c in CONFIGS:
        r = results[c]
        u = f"{r['usd_per_1m_tokens']:.4f}" if r["usd_per_1m_tokens"] else "n/a"
        print(f"{c:>20}  {r['p99_ms']:>9.2f}  {u:>9}  {r['slo_attainment']*100:>5.0f}%  "
              f"{r['migrations_total']:>6}")
    refu = (ref["usd_per_1k_tokens"] or 0) * 1000
    print(f"{'all-fast-ref':>20}  {ref['p99_ms']:>9.2f}  {refu:>9.4f}  "
          f"(n_fast={ref['n_fast']})")
    print(f"[simulate] GEM gap (p99): {gaps['GEM_gap']['p99_improvement_pct']:.1f}%  | "
          f"make-or-break: {gaps['make_or_break']['p99_improvement_pct']:.1f}%")
    print(f"[simulate] temporal-mass fraction (online-over-static ceiling): "
          f"{taxonomy['temporal_mass_fraction']}")
    print(f"[simulate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
