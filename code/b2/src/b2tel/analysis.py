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


def build_windowed(expert_rows: list[dict], phase: str | None = None) -> Windowed:
    """Fold raw expert.jsonl rows into a :class:`Windowed`.

    ``phase`` filters rows by their ``phase`` tag ('prefill'/'decode'/'mixed') so callers can
    build a phase-specific view (Research Plan 2 prefill/decode split). ``None`` = all phases.
    """
    # keep only per-expert rows: a capture dir read globs *.jsonl, which also picks up the
    # sibling expert_mb.*.jsonl micro-batch summary rows (no expert_id) — skip those.
    expert_rows = [r for r in expert_rows if "expert_id" in r and "token_load" in r]
    if phase is not None:
        expert_rows = [r for r in expert_rows if r.get("phase") == phase]
    if not expert_rows:
        raise ValueError("no expert rows" + (f" for phase={phase}" if phase else ""))
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


# --- per-GPU (EP-layout) imbalance: token-load vs activated-expert (METRO) -----------------
# Both the verdict's per-GPU numbers and the logger's per-micro-batch numbers use this SAME
# definition + round-robin layout (expert e -> GPU e mod D), so window-level and
# micro-batch-level imbalance are comparable.

def per_gpu_imbalance(loads: np.ndarray, n_gpus: int) -> tuple[float, float]:
    """Round-robin per-GPU (token_load_imbalance, activated_expert_imbalance) for one layer.

    Each ``max/mean`` over the D GPUs (1.0 = balanced), matching :func:`imbalance_ratio`. The
    activated-expert quantity (distinct experts with load>0 per GPU) is the weight-load /
    decode-cost metric: it can stay skewed even when token-load looks flat at top-k/N_experts.
    """
    if n_gpus <= 1:
        return 1.0, 1.0
    idx = np.arange(loads.shape[0]) % n_gpus
    tok = np.zeros(n_gpus)
    act = np.zeros(n_gpus)
    np.add.at(tok, idx, loads)
    np.add.at(act, idx, (loads > 0).astype(float))

    def _ratio(v: np.ndarray) -> float:
        m = v.mean()
        return float(v.max() / m) if m > 0 else 1.0
    return _ratio(tok), _ratio(act)


def per_gpu_imbalance_series(w: Windowed, n_gpus: int):
    """Per-window per-GPU imbalance, averaged over layers. Returns (times, tok[], act[])."""
    tok_s, act_s = [], []
    for wid in w.window_ids:
        tk, ac = [], []
        for layer in w.layers:
            t, a = per_gpu_imbalance(w.layer_load(wid, layer), n_gpus)
            tk.append(t)
            ac.append(a)
        tok_s.append(float(np.mean(tk)) if tk else 1.0)
        act_s.append(float(np.mean(ac)) if ac else 1.0)
    return w.times, np.array(tok_s), np.array(act_s)


# --- placement-free skew: top-N mass vs a matched balls-in-bins null (the generality verdict)

def topn_mass(loads: np.ndarray, n: int) -> float:
    """Fraction of token mass captured by the top-``n`` experts (placement-free skew)."""
    s = loads.sum()
    if s <= 0:
        return 0.0
    top = np.sort(loads)[::-1][:max(1, n)]
    return float(top.sum() / s)


def topn_mass_skew(w: Windowed, n: int, n_trials: int = 20, seed: int = 0) -> dict:
    """Observed top-``n`` mass vs a **matched** uniform-random (balls-in-bins) null.

    The null throws the *same per-(window,layer) token-slot count* uniformly over the
    ``n_experts`` bins and measures its top-n mass — so the comparison controls for how few
    distinct experts a small batch can even activate. ``ratio`` = observed / null is the
    direction-fixed discriminator for criterion 3 (≥ 2× ⇒ genuinely skewed).
    """
    rng = np.random.default_rng(seed)
    obs, nul = [], []
    p = np.full(w.n_experts, 1.0 / w.n_experts)
    for wid in w.window_ids:
        for layer in w.layers:
            loads = w.layer_load(wid, layer)
            slots = int(round(float(loads.sum())))
            if slots <= 0:
                continue
            obs.append(topn_mass(loads, n))
            draws = rng.multinomial(slots, p, size=n_trials).astype(float)
            nul.append(float(np.mean([topn_mass(d, n) for d in draws])))
    o = float(np.mean(obs)) if obs else 0.0
    u = float(np.mean(nul)) if nul else 0.0
    return {
        "n": n, "n_experts": w.n_experts,
        "observed_topn_mass": o, "null_topn_mass": u,
        "ratio": (o / u) if u > 0 else None,
        "n_samples": len(obs),
    }


# --- hot-streak CDF: once hot, how long does an expert stay hot? (Khuzaima's ask) ----------

def hot_set_mass(loads: np.ndarray, mass_frac: float = 0.8) -> set[int]:
    """Smallest set of top experts whose cumulative load ≥ ``mass_frac`` of the layer total.

    This is the **hot-set** definition for streaks/turnover (top-N by mass, ~80%), not the
    routing top-k — used consistently so Qwen and OLMoE are measured the same way."""
    s = loads.sum()
    if s <= 0:
        return set()
    order = np.argsort(loads)[::-1]
    cum = 0.0
    out: set[int] = set()
    for i in order:
        if loads[i] <= 0:
            break
        out.add(int(i))
        cum += float(loads[i])
        if cum >= mass_frac * s:
            break
    return out


def _window_dt(times: list[float]) -> float:
    """Representative window duration (seconds) for converting streak lengths to time."""
    if len(times) < 2:
        return 1.0
    diffs = np.diff(np.array(times))
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if diffs.size else 1.0


def hot_streak_cdf(w: Windowed, mass_frac: float = 0.8) -> dict:
    """CDF of consecutive-window hot streaks per (layer, expert).

    For each (layer, expert), find maximal runs of consecutive windows in which it is in the
    layer's ~``mass_frac`` hot set; collect every run length. Returns the streak-length
    distribution (in windows and seconds) + median/mean — answering "once hot, how long does
    an expert stay hot?" A median streak > 1 window is criterion 2 of the make-or-break rubric.
    """
    dt = _window_dt(w.times)
    # membership[(layer,e)] across windows in order
    hot_by_window = [
        {layer: hot_set_mass(w.layer_load(wid, layer), mass_frac) for layer in w.layers}
        for wid in w.window_ids
    ]
    streaks: list[int] = []
    for layer in w.layers:
        run: dict[int, int] = {}
        for wi in range(len(w.window_ids)):
            hot = hot_by_window[wi][layer]
            # extend running streaks for still-hot experts; close streaks that ended
            for e in list(run):
                if e not in hot:
                    streaks.append(run.pop(e))
            for e in hot:
                run[e] = run.get(e, 0) + 1
        streaks.extend(run.values())  # streaks still open at capture end (censored)
    arr = np.array(sorted(streaks)) if streaks else np.array([])
    if arr.size:
        xs = arr.tolist()
        cdf = (np.arange(1, arr.size + 1) / arr.size).tolist()
    else:
        xs, cdf = [], []
    return {
        "mass_frac": mass_frac,
        "window_dt_s": dt,
        "n_streaks": int(arr.size),
        "median_windows": float(np.median(arr)) if arr.size else None,
        "mean_windows": float(arr.mean()) if arr.size else None,
        "median_s": float(np.median(arr) * dt) if arr.size else None,
        "p90_windows": float(np.percentile(arr, 90)) if arr.size else None,
        "lengths_windows": xs,
        "cdf": cdf,
    }


# --- micro-batch vs window imbalance (M1 cache-timescale / Lina comparison) ----------------

def microbatch_vs_window(mb_rows: list[dict], w: Windowed, n_gpus: int,
                         phase: str | None = None) -> dict:
    """Compare the imbalance each all-to-all suffers (per-micro-batch, from ``expert_mb.jsonl``)
    against the imbalance a windowed reactive cache sees (from the windowed load ``w``).

    ``w`` should be the phase-matched :class:`Windowed`. If micro-batch ≈ window, the hot set is
    stable across micro-batches and a seconds-granularity cache is justified (Lina's
    per-micro-batch machinery is overkill); if micro-batch ≫ window, the cache chases a moving
    target → motivates prediction (M5) / cite Lina."""
    rows = [r for r in mb_rows if (phase is None or r.get("phase") == phase)]
    # window-level per-GPU imbalance (averaged over layers/windows)
    _, tok_win, act_win = per_gpu_imbalance_series(w, n_gpus)
    win_tok = float(np.mean(tok_win)) if tok_win.size else 1.0
    win_act = float(np.mean(act_win)) if act_win.size else 1.0
    mb_tok = np.array([float(r["tok_imb_mean"]) for r in rows if "tok_imb_mean" in r])
    mb_act = np.array([float(r["act_imb_mean"]) for r in rows if "act_imb_mean" in r])
    mb_tok_p95 = np.array([float(r["tok_imb_p95"]) for r in rows if "tok_imb_p95" in r])
    return {
        "phase": phase, "n_sim_gpus": n_gpus, "n_mb_rows": len(rows),
        "window_tok_imbalance": win_tok, "window_act_imbalance": win_act,
        "microbatch_tok_imbalance_mean": float(mb_tok.mean()) if mb_tok.size else None,
        "microbatch_tok_imbalance_p95": float(mb_tok_p95.mean()) if mb_tok_p95.size else None,
        "microbatch_act_imbalance_mean": float(mb_act.mean()) if mb_act.size else None,
        # >1 ⇒ each all-to-all is more skewed than the window average the cache sees
        "tok_microbatch_over_window": (float(mb_tok.mean()) / win_tok)
        if (mb_tok.size and win_tok > 0) else None,
        "act_microbatch_over_window": (float(mb_act.mean()) / win_act)
        if (mb_act.size and win_act > 0) else None,
    }
