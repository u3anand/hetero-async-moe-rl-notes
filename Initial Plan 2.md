# B2 — OLMoE Expert-Hotness × Tier Profiling on watgpu308

The first runnable milestone of [[Research Plan 2]] (TierShift). Goal: serve a fixed MoE on the heterogeneous node, replay a mixed time-varying request trace, and emit the Q1–Q7 telemetry that answers two questions — does expert hotness drift enough, on the tier gap, to defeat a static expert→tier placement; and does the hot set shift *slower* than an expert can migrate over PCIe? This is "make it serve + measure," not "build the controller." Hardware: [[WatGPU]]. Comparison vs B1: [[Benchmark Comparison]].

OLMoE-1B-7B has **64 experts (8 active/token)** — enough to exhibit skew — and **fits one 46 GB card**, so expert-load capture needs no multi-GPU plumbing.

**Routing granularity is an axis of the benchmark.** Skew shrinks with finer-grained routing ([[GEM]] saw only ~1.5% on Qwen3-30B-A3B's near-uniform 128-expert routing), so the capture spans **Mixtral-8x7B** (8 experts → sharp skew; needs 2 cards / quant just to log routing), **OLMoE-64e** (mid), and **Qwen3-30B-A3B** (128 → flat). The coarse model carries the "skew exists" claim; Qwen3 is the scale-up / realism target.

## Characterize cheaply, validate later

Most of B2 does not need live heterogeneous EP serving — the expert-load distribution is **model + input** determined, not hardware determined:

- **Q1/Q5/Q6** (hotness non-stationarity, reactive-cache locality, domain-sensitivity) = a router hook on **one GPU**, replaying the trace and logging per-expert token counts.
- **Tier gap** = a micro-bench timing one expert's FFN on Ada vs A6000.
- **Migration cost (Q3)** = a micro-bench: expert-weight size × PCIe Ada↔A6000 transfer time, vs. measured hot-set turnover.
- **Static-placement straggler (Q2) + migrate-vs-replicate (Q4)** = offline simulation joining load × tier-cost × migration-cost.

So the first result needs only **single-GPU capture + two micro-benches + offline math.** Live cross-tier EP serving is the last gate (B7), validating the offline model. This is why B2 is ~⅓ the lift of B1.

## Placement configs (simulated, then validated)

Model + trace held constant; only the expert→tier assignment changes.

| Config | Expert→tier rule | Role |
|---|---|---|
| **static-balanced** | round-robin across Ada+A6000 | naive EP default |
| **static-avg-opt** | offline optimum for the *average* trace (hot→Ada) | Aurora-style static optimum |
| **GEM-style** | static speed-aware mapping, one-expert-one-GPU, no online adaptation | closest static hetero baseline |
| **HarMoEny-style** | homogeneous dynamic token redistribution (no tier-speed term) | closest dynamic baseline |
| **CRAFT-style** | cost-aware replication, tier-blind | replication baseline |
| **reactive-LRU-port** | reactive: recently-hot top-K → fast tier, no split | the plain reactive cache |
| **reactive+split** | reactive cache + replicate hottest across tiers, split tokens by speed | the full policy |
| **oracle-dynamic** | per-window optimal given future load | the ceiling |

Two comparisons decide whether the dynamic/split machinery is worth building:
- **GEM-style → reactive+split** — headroom of online over the best static hetero baseline. If ~0, static suffices.
- **reactive-LRU-port → reactive+split** — headroom of splitting over a plain reactive cache.

**Consistent/temporal taxonomy (from GEM):** classify each expert per window as **consistent** (hot in ≳85% of windows — a static map already places it well) or **temporal** (bursty). The fraction of hot-token mass that is *temporal* is the headroom dynamic placement has over static — report it.

## GPU/CPU budget on watgpu308
8× `schoolgpu` = 4 L40S + 2 RTX A6000 + 2 RTX 6000 Ada (46–48 GB, PCIe Gen4, no NVLink). Bring-up needs only **1 Ada + 1 A6000** (capture + micro-benches); live EP validation (B7) uses 2–4. No trainer / rollout agent / udocker → CPU is not a bottleneck here. Grab the node `--exclusive` for clean telemetry. PCIe-not-NVLink makes migration cost real and measurable (Q3).

## Components to reuse
**vLLM** (fork `u3anand/vllm`, already in `code/forks/`) — serve OLMoE/Qwen3-MoE in EP mode + a kernel-free MoE-gate hook logging per-expert token load (shared with B1's Track-A2). · **`b2tel`** (under `code/b2/`) — JSONL writers + offline simulator + charts. · traces: ShareGPT + LMSYS-Chat-1M + a code/math slice, concatenated into time-varying segments. No prime-rl / mini-swe-agent / udocker (those are B1-only).

## Dir layout (`/u3/u3anand/b2/`)
`models/` (HF_HOME) · `traces/` (parquet + mixed replay manifest) · `runs/<config>/<ts>/{telemetry,sim}` · `charts/`. Code in `code/b2/`. Env: `HF_HOME`, `NCCL_SOCKET_IFNAME=eth0` (multi-GPU B7 only).

## Steps (each gated GO/NO-GO)

**0 — Recon** (`salloc -p SCHOOL -w watgpu308 --gres=gpu:1 -t 0:30:00`): `nvidia-smi -L` → index→tier map; pick a fast (Ada) and slow (A6000) CUDA index. **Gate:** map = {4 L40S, 2 A6000, 2 RTX6000 Ada}; indices chosen.

**1 — Env** (`salloc --gres=gpu:2 --cpus-per-task=8 --mem=64G -t 7:00:00`): `bash code/b2/setup.sh` (uv venv + vLLM fork + OLMoE download). **Gate:** vLLM imports on a compute node; `vllm serve allenai/OLMoE-1B-7B-... --dtype bfloat16` answers one request on 1 Ada.

**2 — Trace build** (CPU, ∥ step 1): assemble the mixed manifest — chat + code + math ordered into segments that shift the active domain over wall-clock. **Gate:** manifest replays; segment labels logged (Q6).

**3 — Expert-load capture** (GPU, 1 Ada): serve OLMoE with the router hook on; replay the trace; log per-(layer, expert, window) token load → `expert.jsonl`. **Gate:** per-expert load extractable; imbalance ratio shifts across segments (Q1/Q5/Q6). If flat → expert load is stationary, B2 reports the negative.

**4 — Tier micro-bench** (1 Ada + 1 A6000): time one expert's FFN on Ada vs A6000 across batch sizes → per-tier tokens/s and the fast:slow ratio (expect ~2.4×, larger FP8-Ada vs BF16-Ampere). **Gate:** ratio recorded per batch.

**5 — Migration micro-bench** (1 Ada + 1 A6000): measure expert-weight bytes + PCIe Ada↔A6000 transfer ms; from step 3, measure hot-set turnover. **Gate (Q3):** `migration_ms` vs `turnover_period` recorded → migrate viable iff migration ≪ turnover; else replication-only.

**6 — Offline simulate + charts** (CPU): join `expert.jsonl` × tier-cost × migration-cost × `fleet_cost.json` → simulate the configs, scored on **p99 latency / SLO-attainment + $/token**. **Gate:** under drift, static violates a chosen p99 SLO while reactive holds it; $/token-at-SLO is below the all-fast-GPU reference; the GEM-style→reactive+split and reactive-LRU→reactive+split gaps are reported.

**7 — Live EP validation** *(required; 1 Ada + 1 A6000)*: serve OLMoE EP across both tiers under `static-balanced` (and one more config); confirm measured per-tier busy%/straggler/p99 matches the simulated prediction within a stated tolerance. The result should not rest on simulation alone. **Gate:** live ≈ simulated within tolerance. (Online migration live stays in the controller phase — B7 validates the cost/straggler model only.)

## Instrumentation

Two append-only JSONL streams in `runs/<config>/<ts>/telemetry/`, joined on `(layer_id, expert_id, window_id)`:

**1. `expert.jsonl`** — per (layer, expert, window), from the vLLM MoE-gate hook:
```
window_id, t_start, t_end, layer_id, expert_id, token_load,
tier_assigned{fast|slow|none}, is_replica, segment_label{chat|code|math|mixed},
phase{prefill|decode}, expert_class{consistent|temporal}
```
**2. `system.jsonl`** — ~1 Hz sidecar (live B7 only):
```
per-GPU sm_active%, dram_active% (HBM bw), tier{fast|slow};
cross_tier_alltoall_bytes, per_tier_straggler_wait_s, decode_queue_depth
```
Plus three static artifacts: `tier_cost.json` (fast:slow tokens/s, from B4), `migration_cost.json` (expert bytes, PCIe ms, turnover period, from B5), `fleet_cost.json` ($/hr per tier, for $/token).

| Q | Hook | Derived metric → chart |
|---|---|---|
| Q1 | gate hook, trace replay | per-expert load + imbalance ratio over wall-clock → non-stationarity chart |
| Q2 | offline sim (load × tier × fleet cost) | p99 latency / SLO-attainment + $/token, static vs reactive → SLO-under-drift + cost-at-SLO |
| Q3 | migration micro-bench | migration-ms vs hot-set turnover → the kill-chart |
| Q4 | offline sim | migrate vs replicate vs split + VRAM cost; the GEM-style and reactive-LRU gaps |
| Q5 | gate hook over windows | does keeping *recently*-hot experts resident capture the *next* window's hot set? (reactive hit-rate) |
| Q6 | gate hook × segment | hot-set shift by domain (code/chat/math) → heatmap |
| Q7 | (live B7) sampler | cross-tier all-to-all bytes + HBM-bw per tier |

Reported **p50/p95/p99** over steady windows, as a surface across configs.

## Definition of done
On watgpu308, OLMoE serves a mixed drifting trace, the gate hook emits `expert.jsonl`, the micro-benches emit `tier_cost.json` + `migration_cost.json` + `fleet_cost.json`, and the offline simulator renders Q1–Q7 — headlined by expert-hotness-over-time (Q1), p99-SLO-under-drift + $/token-at-SLO (Q2), the migration-vs-turnover kill-chart (Q3), and the GEM-style / reactive-LRU gaps (Q4). Anchor numbers recorded (imbalance swing, p99 + $/token per config, slow-tier straggler %, migration ms, turnover period, reactive hit-rate, temporal-mass fraction). One live EP run validates the model. The Qwen3-30B-A3B phase reuses the harness.

## Out of scope this phase
- The online TierShift controller (M1–M4) — B2 only profiles.
- Qwen3-30B-A3B / FP8 / multi-GPU EP at scale (next phase).
- Multi-node / sharper Hopper-Blackwell tier gap.
- Live cross-tier migration during serving (B2 measures migration *cost*).

## Top risks → fallbacks
- **Q3: hotness churns faster than PCIe migration** → fall back to tier-aware replication-only; report the timescale as the finding.
- **Dynamic ≈ static (GEM-style → reactive+split ~0)** → report "static suffices"; the regime map stands as the result. The GEM-style baseline + temporal-mass fraction are what settle this.
- **Q1 flat / Q5 low (no locality)** → no reactive premise; B2 is a negative characterization.
- **Routing too uniform (Qwen3 → 1.5%)** → the coarse model carries the skew claim; don't headline Qwen3.
- **Capture confounds** → log prefill/decode separately, keep deterministic settings, and sweep window size (50 ms…5 s) reporting the sensitivity of dwell/churn/temporal-mass.
- **vLLM EP doesn't expose per-expert load** → the gate hook (Track A) is the fix; same hook B1-A2 needs.
- **Trace not inducing drift** → curate sharper domain segments (code-heavy vs chat).

## Sequencing
Spine: 0 → 1 → (2 ∥ 1) → 3 → (4, 5) → 6 → 7. Track A (the vLLM gate hook) is laptop work, written before the cluster window, shared with B1-A2. Runbook: `code/b2/RUNBOOK.md`.
