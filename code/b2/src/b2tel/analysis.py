"""Shared derivations over the captured ``expert.jsonl`` (used by migrate_bench, simulate,
charts). Pure-python / numpy — no GPU, no extra deps beyond numpy.

Vocabulary:
- A **window** is one ``window_id`` from the hook (a wall-clock slice, default 1 s).
- ``load[layer][expert]`` = token_load summed within a window.
- **hot set** of a window/layer = the top-k experts by load.
- **imbalance ratio** = max load / mean load over the experts of a layer (the headline
  non-stationarity signal: it should swing as the workload domain drifts).
- **turnover period** = how long until a layer's hot set substantially changes (Jaccard
  with the reference window drops below a threshold). This is the timescale the migration
  cost (B5) is raced against in the Q3 kill-test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Windowed:
    """Per-window expert load, indexed consistently for downstream math."""

    window_ids: list[int]                       # ordered unique window ids
    times: list[float]                          # representative wall time per window
    layers: list[int]                           # sorted unique layer ids
    n_experts: int
    top_k: int
    # load[w][layer] -> np.ndarray[n_experts]
    load: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)

    def layer_load(self, w: int, layer: int) -> np.ndarray:
        return self.load[w].get(layer, np.zeros(self.n_experts))

    def total_load(self, w: int) -> np.ndarray:
        """Load summed across layers for window ``w`` (global expert view)."""
        acc = np.zeros(self.n_experts)
        for layer in self.layers:
            acc = acc + self.layer_load(w, layer)
        return acc


def build_windowed(expert_rows: list[dict]) -> Windowed:
    """Fold raw expert.jsonl rows into a :class:`Windowed`."""
    if not expert_rows:
        raise ValueError("no expert rows")
    n_experts = max(int(r.get("n_experts", 0)) for r in expert_rows)
    n_experts = max(n_experts, 1 + max(int(r["expert_id"]) for r in expert_rows))
    top_k = max(int(r.get("top_k", 1)) for r in expert_rows)
    layers = sorted({int(r["layer_id"]) for r in expert_rows})

    # accumulate per (window, layer)
    tmp: dict[int, dict[int, np.ndarray]] = {}
    wtime: dict[int, list[float]] = {}
    for r in expert_rows:
        w = int(r["window_id"])
        layer = int(r["layer_id"])
        e = int(r["expert_id"])
        tmp.setdefault(w, {})
        arr = tmp[w].setdefault(layer, np.zeros(n_experts))
        arr[e] += float(r["token_load"])
        wtime.setdefault(w, []).append(float(r.get("t_start", 0.0)))

    window_ids = sorted(tmp)
    times = [float(np.mean(wtime[w])) for w in window_ids]
    return Windowed(
        window_ids=window_ids, times=times, layers=layers,
        n_experts=n_experts, top_k=top_k, load=tmp,
    )


def imbalance_ratio(loads: np.ndarray) -> float:
    """max/mean over experts; 1.0 = perfectly balanced. NaN-safe."""
    s = loads.sum()
    if s <= 0:
        return 1.0
    mean = loads.mean()
    return float(loads.max() / mean) if mean > 0 else 1.0


def imbalance_series(w: Windowed, per_layer: bool = False):
    """Imbalance ratio over windows. Returns (times, ratios) where ratios is a 1-D array
    (mean across layers) or a dict{layer: array} when ``per_layer``."""
    if per_layer:
        out = {layer: [] for layer in w.layers}
        for wid in w.window_ids:
            for layer in w.layers:
                out[layer].append(imbalance_ratio(w.layer_load(wid, layer)))
        return w.times, {k: np.array(v) for k, v in out.items()}
    ratios = []
    for wid in w.window_ids:
        per = [imbalance_ratio(w.layer_load(wid, layer)) for layer in w.layers]
        ratios.append(float(np.mean(per)) if per else 1.0)
    return w.times, np.array(ratios)


def hot_set(loads: np.ndarray, k: int) -> set[int]:
    """Top-k expert ids by load (ignores zero-load experts)."""
    if k <= 0:
        return set()
    order = np.argsort(loads)[::-1]
    return {int(i) for i in order[:k] if loads[i] > 0}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 1.0


def turnover_period(
    w: Windowed, k: int | None = None, threshold: float = 0.5
) -> dict:
    """Estimate how long a hot set stays put. For each window as a reference, walk forward
    until the per-layer-averaged Jaccard with the reference drops below ``threshold``;
    record the elapsed wall time. Returns median/mean turnover seconds + the curve.

    ``k`` defaults to the model's top_k (the natural resident-set size).
    """
    k = k or w.top_k
    ref_hot = {
        wid: {layer: hot_set(w.layer_load(wid, layer), k) for layer in w.layers}
        for wid in w.window_ids
    }
    periods: list[float] = []
    for i, wid in enumerate(w.window_ids):
        t0 = w.times[i]
        crossed = None
        for j in range(i + 1, len(w.window_ids)):
            sims = [
                jaccard(ref_hot[wid][layer], ref_hot[w.window_ids[j]][layer])
                for layer in w.layers
            ]
            if np.mean(sims) < threshold:
                crossed = w.times[j] - t0
                break
        if crossed is not None:
            periods.append(crossed)
    arr = np.array(periods) if periods else np.array([])
    return {
        "k": k,
        "threshold": threshold,
        "median_s": float(np.median(arr)) if arr.size else None,
        "mean_s": float(arr.mean()) if arr.size else None,
        "n_samples": int(arr.size),
        "n_windows": len(w.window_ids),
        # if the hot set never turns over within the capture, the period is ≥ capture span
        "censored": arr.size == 0,
        "capture_span_s": (w.times[-1] - w.times[0]) if len(w.times) > 1 else 0.0,
    }


def classify_experts(
    w: Windowed, k: int | None = None, consistent_frac: float = 0.85
) -> dict:
    """GEM consistent/temporal taxonomy (Initial Plan 2, the core metric).

    For each (layer, expert), the *hot fraction* = fraction of windows in which it is in the
    layer's top-k. **Consistent** = hot in ≥ ``consistent_frac`` of windows (a static average
    places it well — GEM already wins here). **Temporal** = hot in some but fewer windows
    (bursty: heavy when active, invisible to a static average).

    The headline number is ``temporal_mass_fraction``: of all hot-token mass (load counted
    only in the windows where an expert is hot), the share owned by *temporal* experts. To
    first order this is the *ceiling* on how much an online policy (TierShift) can beat the
    best static hetero baseline (GEM). ~0 ⇒ no contribution room; large ⇒ headroom exists.
    """
    k = k or w.top_k
    n_win = len(w.window_ids)
    if n_win == 0:
        return {"temporal_mass_fraction": None, "experts": {}}

    hot_count: dict[tuple[int, int], int] = {}
    hot_mass: dict[tuple[int, int], float] = {}
    for wid in w.window_ids:
        for layer in w.layers:
            loads = w.layer_load(wid, layer)
            hs = hot_set(loads, k)
            for e in hs:
                key = (layer, e)
                hot_count[key] = hot_count.get(key, 0) + 1
                hot_mass[key] = hot_mass.get(key, 0.0) + float(loads[e])

    experts: dict[str, dict] = {}
    temporal_mass = 0.0
    consistent_mass = 0.0
    n_consistent = n_temporal = 0
    for key, cnt in hot_count.items():
        frac = cnt / n_win
        mass = hot_mass[key]
        is_consistent = frac >= consistent_frac
        if is_consistent:
            consistent_mass += mass
            n_consistent += 1
        else:
            temporal_mass += mass
            n_temporal += 1
        experts[f"{key[0]}:{key[1]}"] = {
            "layer": key[0], "expert": key[1], "hot_fraction": frac,
            "hot_mass": mass, "expert_class": "consistent" if is_consistent else "temporal",
        }
    total_mass = temporal_mass + consistent_mass
    return {
        "consistent_frac_threshold": consistent_frac,
        "k": k,
        "n_consistent": n_consistent,
        "n_temporal": n_temporal,
        "consistent_mass": consistent_mass,
        "temporal_mass": temporal_mass,
        "temporal_mass_fraction": (temporal_mass / total_mass) if total_mass > 0 else None,
        "experts": experts,
    }


def reactive_hit_rate(w: Windowed, k: int | None = None) -> dict:
    """Q5 cache-precondition: if we keep the *previous* window's hot set resident, what
    fraction of the *current* window's hot experts are already there? Per-layer-averaged,
    then averaged over windows. High hit-rate ⇒ a reactive (no-forecast) cache works."""
    k = k or w.top_k
    rates: list[float] = []
    for i in range(1, len(w.window_ids)):
        prev, cur = w.window_ids[i - 1], w.window_ids[i]
        per_layer = []
        for layer in w.layers:
            prev_hot = hot_set(w.layer_load(prev, layer), k)
            cur_hot = hot_set(w.layer_load(cur, layer), k)
            if cur_hot:
                per_layer.append(len(prev_hot & cur_hot) / len(cur_hot))
        if per_layer:
            rates.append(float(np.mean(per_layer)))
    arr = np.array(rates) if rates else np.array([])
    return {
        "k": k,
        "mean_hit_rate": float(arr.mean()) if arr.size else None,
        "per_window": rates,
        "times": w.times[1:],
    }
