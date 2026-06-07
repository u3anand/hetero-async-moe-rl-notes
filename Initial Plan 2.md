# B2 — OLMoE Expert-Hotness × Tier Profiling on watgpu308

The first runnable milestone of [[Research Plan 2]] (TierShift). Goal: **serve a fixed MoE on the heterogeneous node**, replay a **mixed, time-varying request trace**, and emit the Q1–Q7 telemetry that shows **expert hotness is non-stationary and interacts with tier heterogeneity** — and answer the **kill-question** (d
oes the hot set churn slower than an expert can migrate over PCIe?). This is "make it serve + measure," not "build the controller." Hardware: [[WatGPU]]. First-milestone comparison vs B1: [[Benchmark Comparison]].

OLMoE-1B-7B has **64 experts (8 active/token)** → enough experts to exhibit skew, and **fits one 46 GB card** → expert-load capture needs no multi-GPU plumbing. (MoE is now *central*, unlike B1 where it was incidental.)

**Routing granularity is now an explicit axis of the benchmark — because [[GEM]] showed it decides whether there's any skew to exploit:** on **Qwen3-30B-A3B** (128 fine-grained experts) GEM got only **~1.5%**, due to *near-uniform routing*; the skew (and our whole premise) lives in **coarser-routing** models. So the capture spans three points: **Mixtral-8x7B** (8 experts, top-2 → sharpest skew; needs 2 cards / quant just to *log routing*), **OLMoE-64e** (mid), **Qwen3-30B-A3B** (128 → flattest, the stress-test for "is there anything here"). The regime map gets a *routing-granularity* dimension. **Do not pitch Qwen3-30B-A3B as the headline evidence** — it may be the *weakest* case; it's the scale-up/realism target, with the coarse model carrying the "skew exists" claim.

## The core design insight — characterize cheaply, validate later

Most of B2 does **not** need live heterogeneous EP serving. The expert-load distribution is **model + input** determined, *not* hardware determined — so:

- **Q1/Q5/Q6 (hotness non-stationarity, reactive-cache locality, domain-sensitivity)** = a router-logit hook on **one GPU**, replaying the trace and logging per-expert token counts over time.
- **Tier gap** = a micro-bench timing one expert's FFN on Ada vs A6000 → per-tier tokens/s.
- **Migration cost (Q3, the kill-test)** = a micro-bench: expert-weight size × PCIe Ada↔A6000 transfer time, vs. measured hot-set turnover rate.
- **Static-placement straggler (Q2) + migrate-vs-replicate (Q4)** = **offline simulation** joining the load-trace × tier-cost × migration-cost.

→ The first, load-bearing result needs only **single-GPU capture + two micro-benches + offline math.** Live cross-tier EP serving is the **last** gate (B7), validating the offline model — not a prerequisite for the headline charts. This is why B2 is ~⅓ the lift of B1.

## The configs — fixed model, vary expert→tier placement (in simulation first, then live)

Controlled comparison: **model + trace held constant; only the expert→tier assignment changes.** The point is to show **no single static placement wins across the drifting workload mix.**

| Config | Expert→tier rule | Layer / role |
|---|---|---|
| **static-balanced** | round-robin experts across Ada+A6000 | the naive EP default |
| **static-avg-opt** | offline-optimal mapping for the *average* trace (hot→Ada, cold→A6000) | Aurora-style static optimum |
| **GEM-style** | static **speed-aware** mapping (faster tier gets proportionally more load), one-expert-one-GPU, no online adaptation | **the closest static hetero baseline** — what we must beat to claim "online matters" |
| **HarMoEny-style** | **homogeneous** dynamic token redistribution to under-utilized GPUs (no tier-speed term) | the closest *dynamic* baseline |
| **CRAFT-style** | cost-aware replication, blind to tiers | reviewers will ask |
| **reactive-LRU-port** | each window, put the **recently**-hottest top-K on Ada (no future, no split) | **Layer 1** — "just copy Expert Buffering to two GPUs" |
| **reactive+split** | reactive-LRU **+ replicate the hottest across tiers and split tokens by speed** | **Layer 2** — the contribution preview |
| **oracle-dynamic** | per-window optimal placement given *future* load | the ceiling everything chases |

**Two gaps matter, not one:**
- **GEM-style (static, speed-aware) → reactive+split** = **the GEM gap** — the headroom of *online* over the best *static* hetero baseline. This is the number a reviewer demands; if ~0, "static (GEM) suffices" (a clean negative). **The win should come specifically from *temporal* experts** (see below) — GEM handles *consistent* ones statically.
- **reactive-LRU-port → reactive+split** = **the make-or-break gap** — headroom of replicate-split over the naive cache-port. If ~0, TierShift is "just Expert Buffering ported."

B2 must size **all three** gaps (static-avg→oracle, GEM→reactive+split, LRU-port→reactive+split). (Live B7 — now required — confirms them on real hardware.)

**Adopt GEM's consistent/temporal expert taxonomy as a core metric.** Classify each expert per window as **consistent** (hot in ≳85% of windows — caught by a static average, so GEM already places it well) vs **temporal** (bursty — hot in only a fraction of windows, but heavy when active). **The fraction of hot-token mass that is *temporal* is, to first order, the headroom of online over static** — i.e. the ceiling on how much TierShift can beat GEM. Report it; it's the cleanest single number that says whether the contribution exists.

## GPU/CPU budget on watgpu308
8× `schoolgpu` = 4 L40S + 2 RTX A6000 + 2 RTX 6000 Ada (46–48 GB, PCIe Gen4, **no NVLink**). Bring-up needs only **1 Ada + 1 A6000** (capture + micro-benches); the live EP validation (B7) uses 2–4. **No trainer, no rollout agent, no udocker reward** → CPU is not a bottleneck here (unlike B1). Grab the node `--exclusive` for clean telemetry anyway. PCIe-not-NVLink is *the point* — it makes migration cost real and measurable (Q3).

## Components to reuse (don't reinvent)
**vLLM** (our fork `u3anand/vllm`, already in `code/forks/`) — serve OLMoE/Qwen3-MoE in EP mode; **add a kernel-free hook at the MoE gate to log per-expert token load** (the *same* router-logit hook B1's Track-A2 needs — share it). · **`b2tel`** (new, under `code/b2/`) — JSONL writers + the offline simulator + charts. · request traces: **ShareGPT** + **LMSYS-Chat-1M** + a **code/math slice** (e.g. a HumanEval/GSM8K sample), concatenated into time-varying segments. **No prime-rl, no mini-swe-agent, no udocker** — those are B1-only.

## Dir layout (`/u3/u3anand/b2/`)
`models/` (HF_HOME, OLMoE → Qwen3-30B-A3B) · `traces/` (ShareGPT/LMSYS/code-math parquet + the mixed replay manifest) · `runs/<config>/<ts>/{telemetry,sim}` · `charts/`. Our code in `code/b2/` (notes repo). Global env: `HF_HOME`, `NCCL_SOCKET_IFNAME=eth0` (only if multi-GPU B7).

## Steps (each gated GO/NO-GO)

**0 — Recon** (`salloc -p SCHOOL -w watgpu308 --gres=gpu:1 -t 0:30:00`): `nvidia-smi -L` → index→tier map; pick a fast (Ada) and slow (A6000) CUDA index. **Gate:** map = {4 L40S, 2 A6000, 2 RTX6000 Ada}; one Ada + one A6000 index chosen.

**1 — Env** (`salloc -p SCHOOL -w watgpu308 --gres=gpu:2 --cpus-per-task=8 --mem=64G -t 7:00:00`): `bash code/b2/setup.sh` (conda env + vLLM fork + OLMoE download). **Gate:** vLLM imports on a compute node; `vllm serve allenai/OLMoE-1B-7B-...` answers one request on 1 Ada (BF16).

**2 — Trace build** (CPU, ∥ step 1): assemble the mixed replay manifest — ShareGPT/LMSYS chat + a code slice + a math slice, ordered into **segments** that shift the active domain over wall-clock (chat → code → math → mixed). **Gate:** manifest replays; segment boundaries logged (the Q6 domain labels).

**3 — Expert-load capture** (GPU, 1 Ada): serve OLMoE with the router hook on; replay the mixed trace; log **per-(layer, expert, time-window) token load** → `expert.jsonl`. **Gate (the headline):** per-expert load extractable; **imbalance ratio visibly shifts across segments** (Q1/Q5/Q6). If it *doesn't* move → B2 reports "expert load is stationary," and TierShift has no premise (publishable negative; pivot signal).

**4 — Tier micro-bench** (1 Ada + 1 A6000): time a single expert's FFN forward (and a small EP all-to-all) on Ada vs A6000 across batch sizes → **per-tier tokens/s** and the **fast:slow expert-compute ratio** (expect ~2.4× compute, larger if FP8-on-Ada vs BF16-on-Ampere). **Gate:** ratio recorded per batch size.

**5 — Migration micro-bench** (1 Ada + 1 A6000) — *the kill-test*: measure one expert's weight tensor size + **cudaMemcpy/PCIe transfer time Ada↔A6000**; from step 3, measure **hot-set turnover rate** (how many windows until the top-K hot set changes by >X%). **Gate (Q3):** `migration_ms` vs `turnover_period` recorded → **migrate is viable iff migration ≪ turnover**; else fall back to **replication-only** (still novel vs homogeneous CRAFT). This decision shapes the whole controller.

**6 — Offline simulate + charts** (CPU): join `expert.jsonl` × tier-cost (step 4) × migration-cost (step 5) → simulate the configs. Score each on the **objective**, not just straggler: derive **p99 token latency** (from per-tier finish times under each placement) and **$/token** (assign a $/hr to each tier via a small `fleet_cost.json`, divide fleet-cost by tokens served). Emit Q1–Q7 charts. **Gate:** under drift, **static placement violates a chosen p99 SLO** while reactive holds it (Q2 as an *SLO* chart, not just straggler); the **$/token-at-fixed-SLO is below the all-fast-GPU reference**; and the **reactive-LRU-port → reactive+split** headroom is non-trivial (Q4). Headline charts render.

**7 — Live EP validation** *(REQUIRED, not optional — closes the "cute simulator" loophole; 1 Ada + 1 A6000)*: actually serve OLMoE EP across the two tiers under `static-balanced` (and one more config), and confirm the **measured** per-tier busy%/straggler/p99 matches the **simulated** prediction **within a stated tolerance**. A serious systems result cannot rest on offline simulation alone; one real EP run that validates the model is the minimum bar. **Gate:** live ≈ simulated within tolerance → the offline regime-map is trustworthy. (Full *online migration* live still deferred to the controller phase — B7 only validates the cost/straggler model.)

## Instrumentation

Two append-only JSONL streams in `runs/<config>/<ts>/telemetry/`, joined on `(layer_id, expert_id, window_id)` and one monotonic clock:

**1. `expert.jsonl`** — per (layer, expert, time-window), from the vLLM MoE-gate hook:
```
window_id, t_start, t_end, layer_id, expert_id, token_load,
tier_assigned{fast|slow|none}, is_replica, segment_label{chat|code|math|mixed},
phase{prefill|decode}, expert_class{consistent|temporal}   # phase split + GEM taxonomy
```
**2. `system.jsonl`** — ~1 Hz sidecar (only populated in live B7):
```
per-GPU sm_active%, dram_active% (HBM bw), tier{fast|slow};
cross_tier_alltoall_bytes, per_tier_straggler_wait_s, decode_queue_depth
```
Plus three static artifacts: `tier_cost.json` (fast:slow tokens/s per batch size, from B4), `migration_cost.json` (expert weight bytes, PCIe ms, turnover period, from B5), and `fleet_cost.json` (a $/hr per tier — e.g. cloud list price for L40S vs A6000 — so the sim can compute **$/token** and the cost-at-SLO story).

| Q | Hook | Raw fields | Derived metric → chart |
|---|---|---|---|
| Q1 | gate hook, trace replay | `expert.jsonl token_load` | per-expert load + **imbalance ratio over wall-clock** → **the headline non-stationarity chart** |
| Q2 | offline sim (load × tier-cost × fleet-cost) | `expert.jsonl` + `tier_cost.json` + `fleet_cost.json` | **p99 latency / SLO-attainment + $/token** under static vs reactive placement → SLO-violation-under-drift chart + cost-at-SLO bar (straggler is the mechanism behind it) |
| Q3 | migration micro-bench | `migration_cost.json` | **migration-ms vs hot-set-turnover-period** → **the kill-chart** |
| Q4 | offline sim | `expert.jsonl` + costs | migrate-vs-replicate-vs-split payoff + VRAM cost of K replicas → bar; **the reactive-LRU-port → reactive+split headroom (make-or-break)** |
| Q5 | gate hook over windows | `expert.jsonl` | **cache-precondition** — does keeping *recently*-hot experts resident capture the *next* window's hot set? (reactive hit-rate, no forecasting) → line |
| Q6 | gate hook × segment label | `expert.jsonl segment_label` | hot-set shift by domain (code/chat/math) → heatmap; reactive control signal |
| Q7 | (suppl., live B7) sampler | `system.jsonl` | cross-tier all-to-all bytes + HBM-bw per tier → series |

Reported **p50/p95/p99** over steady windows, as a surface across the three placement configs.

## Definition of done
On watgpu308, OLMoE serves a **mixed drifting trace**, the gate hook emits `expert.jsonl`, the micro-benches emit `tier_cost.json` + `migration_cost.json` + `fleet_cost.json`, and the offline simulator renders Q1–Q7 — headlined by **expert-hotness-over-time vs static placement** (Q1), the **p99-SLO-violation-under-drift + $/token-at-SLO** chart (Q2, the objective), the **migration-cost vs hotness-timescale kill-chart** (Q3), and the **reactive-LRU-port → reactive+split headroom** (Q4, the make-or-break). Anchor numbers recorded (imbalance-ratio swing, **p99 latency + $/token per config**, slow-tier straggler %, expert-migration ms over PCIe, hot-set turnover period, reactive cache hit-rate, **and the Layer-1→Layer-2 gap**). This proves the harness + tells us whether TierShift (a) holds a p99 SLO at lower $/token than all-fast, and (b) is more than an Expert-Buffering port; the Qwen3-30B-A3B phase reuses it.

## Out of scope this phase (deferred)
- The **online TierShift controller** (placement/migration policy) — B2 only *profiles* and shows static mis-fits; M1–M4 come next.
- Qwen3-30B-A3B / FP8 / multi-GPU EP at scale (next phase, harness unchanged).
- Multi-node / sharper Hopper-Blackwell tier gap (sponsor nodes / AWS).
- Live cross-tier migration *during* serving (B2 measures migration *cost*; it doesn't yet migrate online).

## Top risks → fallbacks
- **Make-or-break (Q4): reactive-LRU-port ≈ reactive+split** → TierShift is "just Expert Buffering on two GPUs," no contribution. Detect early; if the split lever adds nothing, the honest finding is "naive expert-LRU-to-fast-tier suffices" (weaker, still real). **This is the single most important thing B2 must measure.**
- **Kill-risk (Q3): turnover ≪ migration** → fall back to **replication-only** (tier-aware replication still beats homogeneous CRAFT); report the timescale as the bounding finding.
- **Risk: load is stationary (Q1 flat) / no locality (Q5 low)** → no cache-precondition; B2 becomes a negative-result characterization, pivot back to B1.
- vLLM EP doesn't expose per-expert load → the gate hook (Track A) is the fix; it's the *same* hook B1-A2 needs (logits at `self.gate(...)`), so reuse.
- 1-card OLMoE EP is artificial → capture is single-GPU (valid: routing is HW-independent); the *real* tier claim is the multi-GPU phase. Treat OLMoE numbers as harness-proving anchors.
- **GEM-overlap (the novelty dart):** "TierShift = GEM + online + replication." → defense is the **GEM-style → reactive+split** gap + the **temporal-token-mass** fraction; if both are large the contribution is real, if ~0 report "static (GEM) suffices." This is *why* GEM-style is a required baseline, not just a citation.
- **Routing too uniform (per GEM, Qwen3 → 1.5%)** → the coarse model (Mixtral-class) carries the "skew exists" claim; don't headline the fine-grained model.
- **Capture confounds (reviewer-sniping, advisor Risk 3):** routing isn't *perfectly* HW-independent in practice — **log prefill and decode separately** (`phase` field; they have different load profiles), keep **deterministic settings** (fixed seed/batch where possible), and **sweep window size** (50 ms…5 s) reporting the **sensitivity** of dwell/churn/temporal-mass rather than a single cherry-picked window.
- Trace not inducing drift → curate sharper domain segments (code-heavy vs chat) per the measured domain-sensitivity literature (Imbalance Ratio 1.43–2.28).

## Sequencing
Spine: 0 → 1 → (2 ∥ 1) → 3 → (4, 5) → 6 → 7(optional). Track A (the vLLM gate hook) can be written on the laptop **before** the cluster window — and it is shared with B1-A2, so it's not wasted regardless of which direction wins. Runbook: `code/b2/RUNBOOK.md`.
