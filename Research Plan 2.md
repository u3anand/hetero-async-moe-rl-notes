# Research Plan 2

**TierShift — an online, multi-tier *expert cache* for MoE serving on heterogeneous GPUs.**

Treat expert placement as a **3-level cache over a memory/compute hierarchy — fast GPU → slow GPU → CPU** — and manage it online: keep the **hot** experts resident on the **fast GPU**, the **warm** ones on the **slow GPU** (which *still computes* — a "miss" is a straggler, not a stall), and spill the **cold** tail to **CPU** (storage, fetched on demand). As the hot set drifts with the workload, the controller **promotes / demotes / migrates / replicates** experts across the three tiers — *on top of the existing expert-parallel all-to-all communication* — to meet a **p99 latency target at lower $/token**. A characterization phase (**B2**) measures whether this is worth doing; the online controller follows if B2 says yes.

This unifies the two anchor mechanisms: **Expert Buffering** ([[Toward-Efficient-MoE-Inference]]) is the GPU↔CPU *memory* cache (our bottom two levels); **tier-aware placement** ([[GEM]], [[Aurora]]) is the fast↔slow-GPU *compute* split (our top two levels). TierShift is the **single online cache that spans both** — with a computing middle tier and a replicate-and-split lever neither has.

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

**TierShift's cell:** an *online, multi-tier expert cache* (**fast GPU → slow GPU → CPU**) — tier-aware placement + migration + **replicate-and-split** across a real fast/slow **compute** gap with a **computing middle tier**, expert movement **overlapped with the EP all-to-all**, under a **p99-SLO + $/token** objective. It fuses Expert Buffering's memory cache (bottom levels) with GEM/Aurora-style tier placement (top levels), made online.

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

An online controller that **reacts to measured per-expert load** (no workload forecast) and manages a **3-level expert cache — fast GPU / slow GPU / CPU** — under migration-cost + VRAM budgets, to hold a p99 SLO at minimum $/token. It self-adapts to any traffic, with no workload model to tune.

**The cache hierarchy (why three levels):**

| Tier | Holds | On a "miss" | Capacity |
|---|---|---|---|
| **0 — fast GPU** (H200/Ada) | the **hot** experts | — (resident) | small (the cache budget) |
| **1 — slow GPU** (Ada/A6000) | the **warm** experts | served **slower** (a *straggler*, never a stall) — it still computes | larger |
| **2 — CPU RAM** | the **cold** tail | **fetched** over PCIe before compute (Expert Buffering's stall) | spill / unbounded |

The middle tier is the key difference from a pure memory cache: a non-fast expert isn't a stall, it just runs slower — so demotion is cheap and safe. The CPU tier only matters when even the slow GPU can't hold all experts (large / fine-grained models) — so **the same design scales from a 2-tier setup (Mixtral fits in GPU) to a 3-tier one (DeepSeek-256 spills to CPU).**

| # | Mechanism | What it does |
|---|---|---|
| M1 | **Multi-tier reactive cache** | promote recently-hot experts toward the fast GPU, demote cooling ones down (slow GPU → CPU), with hysteresis to avoid thrash. Levels 0–1 are *compute* tiers (load-balancing); level 2 is the *memory* spill |
| M2 | **Replicate-and-split** | for the hottest experts, hold copies on the fast **and** slow GPU and split their tokens by speed under a VRAM budget — the lever GEM (one-expert-one-GPU) and homogeneous replication cannot express |
| M3 | **Tier-gap-aware policy** | decide *promote / demote / replicate / leave* per expert from measured load × tier-gap × migration-cost |
| M4 | **Movement over the all-to-all** | promotions/migrations move expert weights over the **same interconnect the EP all-to-all uses** — the controller must schedule them to **overlap/hide under** the dispatch-combine traffic (and split a replicated expert's tokens in the all-to-all). The open systems bet; validated live (B7). Borrow [[Aurora]]'s comm scheduling here if contention bites |
| M5 (optional) | **Prefetch** | move a rising expert a beat early to hide migration latency — only if B2 shows it is needed (unnecessary when migration ≪ window) |

**Evaluation.** Two framings of the same runs: *(i) fixed fleet → SLO-attainment + throughput; (ii) fixed SLO → $/token (the min-cost fleet that still meets p99).* The **central comparison** is the naive reactive cache (hot→fast, no split) vs. the full **replicate-and-split** policy — the contribution is the gap between them; if it is ~0, "static/naive suffices" and we report that.

Baselines: **static-avg-opt** (offline optimum for the average trace), **GEM-style** (static speed-aware mapping), **HarMoEny-style** (homogeneous dynamic redistribution), **CRAFT-style** (tier-blind replication), **round-robin EP**, and **all-fast-GPU** (the expensive SLO-meeting reference for the cost story). Ablate baseline → +M1 → +M2 → +M3. Success criteria: (1) $/token at fixed SLO below all-fast; (2) p99 SLO held under drift where static violates it; (3) replicate-and-split beats the naive cache *and* the closest dynamic neighbor.

## 4. How we'll use WatGPU

**Tier pair (current):** **H200 (fast) + RTX 6000 Ada (slow)** on the ALL partition — a real architectural gap (~3× compute, ~5× memory bandwidth → biggest in the bandwidth-bound *decode* phase). These sit on **different nodes**, so live EP across them is **cross-node** (all-to-all + migration over the inter-node network) — harder, but realistic. Two cleaner alternatives: (a) **watgpu308** (4× L40S + 2× A6000 + 2× Ada) — a heterogeneous cluster *in one box* (Ada fast + A6000 slow, PCIe Gen4, no NVLink) → clean single-node EP, the canonical setup when it's up; (b) **two identical GPUs on one node + clock/power-cap one** to emulate the slow tier (GEM-style emulated variability) → same-node, controllable gap. See [[WatGPU]]. **Measure the tier gap via end-to-end serving (decode tok/s), not an isolated expert-FFN** (which is too small to expose it).

**Models — validate across the MoE-granularity axis (all three matter).** The cache only pays where routing is skewed, and skew shrinks with finer-grained experts (GEM saw only ~1.5% on Qwen3-30B-A3B's near-uniform 128-expert routing). So we test the whole spectrum:

| Granularity | Model | Role |
|---|---|---|
| **Coarse** (few, big experts) | **Mixtral-8×7B** (8 experts, top-2) | sharpest skew — carries the "skew exists / cache pays" claim |
| **Mid** | **OLMoE-1B-7B** (64, top-8) | bring-up + the B2 headline (✓ imbalance 2.3–8.0) |
| **Fine-grained** (many, small) | **Qwen3-30B-A3B / DeepSeek** (128–256) | the stress test — does the cache survive ~uniform routing, or fall back to tier-split? |

The point is a **map across granularity**: where does the multi-tier cache help, and where does it degrade to GEM-style static tier-split? A clean "stops paying past ~N experts" is itself a result.

Deferred: 16-GPU / multi-node at scale, online migration at scale, and a cloud run for a controlled eval with a stable tier shape.

---

## Workload

| Dim            | Spec                                                                                                                                                                              |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model          | **Mixtral-8×7B** (coarse) → **OLMoE-1B-7B** (mid, bring-up) → **Qwen3-30B-A3B / DeepSeek** (fine) — full granularity sweep                                                        |
| Task           | inference serving only — no fine-tuning, no RL                                                                                                                                    |
| Request stream | trace replay: ShareGPT + LMSYS-Chat-1M + HumanEval + GSM8K, mixed and time-varying (drifting chat→code→math)                                                                      |
| Stack          | vLLM in expert-parallel mode; experts placed across a 3-tier cache; telemetry + migration hooks in the fork                                                                       |
| Heterogeneity  | fast GPU (**H200** / RTX 6000 Ada) → slow GPU (RTX 6000 Ada / A6000) → CPU spill; PCIe Gen4 (intra-node) or inter-node network                                                    |
| Precision      | BF16 (slow tier has no FP8); FP8-on-fast measured as a tier-gap amplifier                                                                                                         |
| Control        | **3-tier reactive cache (fast GPU / slow GPU / CPU)**, no prediction; promote/demote/replicate/split under VRAM + PCIe budgets; expert movement overlapped with the EP all-to-all |

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
