# Research Plan 2

**TierShift — online tier-aware expert placement & migration for MoE serving on heterogeneous GPUs.**

Serve a fixed MoE model on a heterogeneous GPU fleet (a fast tier + a cheaper slow tier) and meet a **p99 latency target at lower $/token** by dynamically **placing, replicating, and migrating experts across tiers** as the hot set shifts with the workload. A characterization phase (**B2**) measures whether this is worth doing; the online controller follows if B2 says yes.

> Serving-focused alternative to [[Research Plan]] (AutoPD-RL) — no RL, no training-to-convergence, feasible on the WatGPU node, which it shares. First-milestone comparison: [[Benchmark Comparison]].

---

## 1. Motivation

MoE now dominates open-weight LLMs (DeepSeek, Qwen3, Mixtral, Kimi). At serving scale, experts are spread across GPUs via **expert parallelism (EP)**, and two facts hold: **expert load is heavily skewed, and it is non-stationary.** On real traces (ShareGPT, LMSYS-Chat-1M; Mixtral-8×7B, Phi-3.5-MoE) expert popularity shifts with the request mix — code requests light up one specialized subset, chat a broader different one — with the imbalance ratio swinging ~1.43→2.28. A heavily-used ("hot") expert creates a **straggler**: the GPU holding it stalls everyone behind the all-to-all barrier.

On a **heterogeneous** fleet this skew collides with hardware. A fast tier (Ada, FP8) and a slow tier (Ampere, BF16) differ ~2.4–4× in compute. A hot expert on the slow tier is a double penalty; a fast tier holding only cold experts is wasted. The right placement is **hotness-aware *and* tier-aware** — and because hotness moves, the right placement moves too. **That moving placement is what TierShift manages.**

**The objective is a latency/cost target.** The reason to run a *mixed* fleet (mostly cheap slow GPUs + a few fast ones) rather than all-fast-GPUs is cost. So the goal is to **meet a p99 latency SLO at minimum $/token** — serve most experts on cheap slow GPUs, keep the hot ones fast enough to hold the SLO. We measure **p99 / SLO-attainment, $/token, throughput, and per-tier straggler**; we make **no model-quality claims** (we serve a fixed checkpoint).

### How TierShift relates to existing work

The literature splits three ways, each missing one axis TierShift needs:

| Neighbor | ID | Has | Missing (vs TierShift) |
|---|---|---|---|
| [[GEM]] | 2605.19945 | GPU-variability-aware expert→GPU mapping; same straggler motivation | static (mapped once); ~10–25% *intra-generation* variability, not a 2–4× tier gap; one-expert-one-GPU (no split) |
| [[Aurora]] | 2410.17043 | optimal MoE placement + comm scheduling on hetero GPUs | static/offline; assumes the traffic matrix is known a priori |
| [[MegaScale-Infer]] | 2504.02263 | MoE serving across hetero tiers (production scale) | static placement + greedy historical replication; no online migration |
| [[HarMoEny]] | 2506.12417 | online token redistribution + async expert prefetch | homogeneous (rebalances to *idle*, not *faster*, GPUs); throughput/TTFT objective |
| CRAFT / PROBE | 2603.28768 / 2602.00509 | online replication / predictive prefetch | homogeneous — no tier-speed model, no cross-tier migration |
| [[Toward-Efficient-MoE-Inference]] | NeurIPS'24 | Expert Buffering — reactive cache of hot experts (the locality result we build on) | memory caching GPU↔CPU (stall-on-miss); not compute-tier load-balancing |
| [[Aurora]]/[[Helix]]/[[HexGen-2]] | — | hetero serving (some dense, some MoE) | dense, or static, or no expert-tier control |

**TierShift's cell:** *online* tier-aware placement + migration + **replicate-and-split** across a real fast/slow **compute** gap, under a **p99-SLO + $/token** objective.

**Closest neighbor — GEM.** GEM does *static*, variability-aware expert→GPU mapping for the ~10–25% manufacturing/DVFS spread among *identical* GPUs. TierShift targets a *2.4–4× architectural* gap, adapts *online*, and can **split a single hot expert across tiers** — which GEM (one-expert-one-GPU) cannot. The magnitude is why: at GEM's ~10–25% a misplaced hot expert costs little, so a one-time map suffices; at 2.4–4× it is a 240–400% penalty, so online correction pays for itself and splitting an over-one-GPU expert becomes necessary.

## 2. B2 — characterize first

Before building the controller, measure how expert hotness actually moves on a real serving stream, on real heterogeneous tiers, and decide **which regime heterogeneous MoE serving is in:**

> **static-sufficient · replication-friendly · migration-friendly** — as a function of (hotness drift × tier gap × migration cost × VRAM budget).

This is useful whatever the answer: a negative ("static suffices for this model/workload") is a result that bounds the design space. Two questions set the regime: (1) does hotness drift, on the tier gap, defeat a static placement? (2) does the hot set shift *slower* than an expert can migrate over PCIe? If experts churn faster than they can move, the regime is replication-only (or static), not migration.

| #   | Question | Determines |
| --- | --- | --- |
| Q1  | How skewed and non-stationary is per-expert load? (per-layer load over time + across domains) | whether the hot set moves enough to defeat static placement |
| Q2  | Under a static placement, how much slow-tier straggler / fast-tier idle does drift cause? | sizes the win |
| Q3  | Migration cost vs. hotness timescale (expert weight size, PCIe transfer, hot-set turnover) | the kill-question — is migration faster than the churn? |
| Q4  | Replicate vs. migrate payoff + VRAM cost of replicas | which mechanism wins, and the memory budget |
| Q5  | Reactive-cache locality — does keeping *recently*-hot experts resident capture the *next* window's hot set? | whether a reactive (no-forecast) controller works |
| Q6  | Workload-mix sensitivity — does the hot set shift with request domain (code/chat/math)? | the reactive control signal |
| Q7  | (suppl.) cross-tier all-to-all bytes + per-tier HBM-bw + queue depths | PCIe/queue pressure |

Reported as p50/p95/p99 over steady-state windows. **Success = the regime is identified with evidence**; only the migration-friendly regime justifies the full migrating controller.

## 3. The system — TierShift

An online controller that **reacts to measured per-expert load** (no workload forecast) and reallocates experts across tiers under migration-cost + VRAM budgets, to hold a p99 SLO at minimum $/token. It is a **reactive cache**, like a CPU cache or Expert Buffering — it keeps recently-hot experts on the fast tier and self-adapts to any traffic, with no workload model to tune.

| # | Mechanism | What it does |
|---|---|---|
| M1 | **Reactive tier-cache** | keep recently-hot experts on the fast tier, demote cold ones to the slow tier, with hysteresis to avoid thrash. The slow tier still *computes* — this is load-balancing across two compute tiers, not memory caching |
| M2 | **Replicate-and-split** | for the hottest experts, hold copies on both tiers and split their tokens by speed under a VRAM budget. This is the lever GEM (one-expert-one-GPU) and homogeneous replication cannot express |
| M3 | **Tier-gap-aware policy** | decide *migrate vs. replicate vs. leave* per expert from measured load × tier-gap × migration-cost |
| M4 (optional) | **Prefetch** | move a rising expert a beat early to hide migration latency — only if B2 shows it is needed (unnecessary when migration ≪ window) |

**Evaluation.** Two framings of the same runs: *(i) fixed fleet → SLO-attainment + throughput; (ii) fixed SLO → $/token (the min-cost fleet that still meets p99).* The **central comparison** is the naive reactive cache (hot→fast, no split) vs. the full **replicate-and-split** policy — the contribution is the gap between them; if it is ~0, "static/naive suffices" and we report that.

Baselines: **static-avg-opt** (offline optimum for the average trace), **GEM-style** (static speed-aware mapping), **HarMoEny-style** (homogeneous dynamic redistribution), **CRAFT-style** (tier-blind replication), **round-robin EP**, and **all-fast-GPU** (the expensive SLO-meeting reference for the cost story). Ablate baseline → +M1 → +M2 → +M3. Success criteria: (1) $/token at fixed SLO below all-fast; (2) p99 SLO held under drift where static violates it; (3) replicate-and-split beats the naive cache *and* the closest dynamic neighbor.

## 4. How we'll use WatGPU

**watgpu308** — 8 GPUs (4× L40S + 2× RTX A6000 + 2× RTX 6000 Ada) — is a heterogeneous cluster in a box: a **fast tier** (Ada, FP8-capable) and a **slow tier** (Ampere A6000, BF16), a real ~2.4× gap, PCIe Gen4, no NVLink (so migration cost is real and measurable). See [[WatGPU]].

**Initial phase:** stand up EP MoE serving across both tiers on **OLMoE-1B-7B** (64 experts — enough to exhibit skew, fits one card), replay a mixed chat/code/math trace, and emit Q1–Q7. Because routing skew shrinks with finer-grained experts (GEM saw only ~1.5% on Qwen3-30B-A3B's near-uniform 128-expert routing), the capture spans **Mixtral-class (sharp skew) → OLMoE (mid) → Qwen3-30B-A3B (flat)**; the coarse model carries the "skew exists" claim and Qwen3 is the realism/scale-up target, not the headline.

Deferred: 16-GPU / multi-node, online migration at scale, and a cloud run for a controlled eval with a stable 2-tier shape.

---

## Workload

| Dim | Spec |
| --- | --- |
| Model | Mixtral-class / OLMoE-1B-7B (bring-up) → Qwen3-30B-A3B (scale-up) |
| Task | inference serving only — no fine-tuning, no RL |
| Request stream | trace replay: ShareGPT + LMSYS-Chat-1M + a code/math slice, mixed and time-varying |
| Stack | vLLM in expert-parallel mode, experts pinned per-tier; telemetry + migration hooks in the fork |
| Heterogeneity | fast tier (L40S / RTX 6000 Ada) vs slow tier (RTX A6000); PCIe Gen4, no NVLink |
| Precision | BF16 (slow tier has no FP8); FP8-on-Ada measured as a tier-gap micro-bench |
| Control | reactive tier-cache (no prediction) + replicate/split under VRAM + PCIe budgets |

## Out of scope
Model-quality claims (fixed checkpoint) · training/RL · new MoE architecture or router · dense-LLM hetero serving (Helix/HexGen-2) · attention/expert tier-splitting (HeterMoE/MegaScale) · homogeneous EP load-balancing (CRAFT/EPLB) · multi-node scale this phase.

## Top risks → fallbacks
- **Migration too slow (Q3):** hotness churns faster than PCIe migration → fall back to tier-aware replication-only; report the timescale as the finding.
- **Routing too uniform:** fine-grained models (Qwen3) show little skew → the coarse model carries the claim; don't headline Qwen3.
- **No win over static (GEM):** if dynamic ≈ static, report "static suffices" — the regime map is the result either way.
- **EP doesn't expose per-expert load / hot-swap:** fork vLLM to emit per-expert counts + a migrate hook.
- **Cross-tier all-to-all over PCIe dominates:** measure it (Q7) as a real hetero cost.

## Related notes
`papers/`: [[GEM]] · [[Aurora]] · [[MegaScale-Infer]] · [[HarMoEny]] · [[Toward-Efficient-MoE-Inference]] · [[SwapMoE]] · [[HeterMoE]] · [[Helix]] · [[HexGen-2]] · [[HexAGenT]]. Sibling plan: [[Research Plan]] (AutoPD-RL). First-milestone detail: [[Initial Plan 2]] · [[Benchmark Comparison]]. Cluster: [[WatGPU]].
