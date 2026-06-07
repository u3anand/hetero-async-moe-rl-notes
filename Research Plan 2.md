# Research Plan 2

**TierShift — online tier-aware expert placement & migration for MoE serving on heterogeneous GPUs.**

A workload-characterization phase (B2) that profiles how MoE **expert hotness** moves over a serving stream and shows that **static expert→tier placement is mis-sized for a moving target** on heterogeneous hardware — then an online controller that places, migrates, and replicates experts across fast/slow GPU tiers to keep the fast tier loaded with hot experts and kill slow-tier stragglers (the system contribution).

> **Alternative direction to [[Research Plan]] (AutoPD-RL), not a replacement.** Proposed 2026-06-07 following Benson's steer toward *inference over RL* (less compute, no convergence claim, feasible on WatGPU). Reuses the same node and much of the same telemetry philosophy. Verified landscape in `papers/` ([[Aurora]], [[MegaScale-Infer]], [[HeterMoE]], [[Helix]], [[HexGen-2]], [[HexAGenT]]); the wedge survived deep reads of the static-placement owners *and* the dynamic-homogeneous cluster (PROBE/CRAFT/Patterns-behind-Chaos). Compare the two plans' first milestones in [[Benchmark Comparison]].

---

## 1. Motivation

MoE now dominates open-weight LLMs (DeepSeek, Qwen3, Mixtral, Kimi). At serving scale, experts are spread across GPUs via **expert parallelism (EP)**, and a hard, well-documented fact emerges: **expert load is heavily skewed *and* non-stationary**. Measured on real traces (ShareGPT, LMSYS-Chat-1M; Mixtral-8×7B, Phi-3.5-MoE), expert popularity *"changes with semantic transitions"* and the **Imbalance Ratio swings 1.43→2.28** as the request mix shifts; code-generation requests light up a specialized expert subset while general chat activates a broader, different set. Hot experts create **stragglers** — the GPU holding an overloaded expert stalls everyone behind the all-to-all.

On a **heterogeneous** cluster this skew collides with hardware: a fast tier (high FLOPS/BW) and a slow tier (older Ampere) differ ~2.4× in compute. If a *hot* expert lands on the *slow* tier, it is a double penalty; if the fast tier holds only *cold* experts, its capability is wasted. The right placement is **expert-hotness-aware *and* tier-aware** — and because hotness moves, the right placement **moves too**.

**Why the existing work doesn't close this.** Each neighbor gives up exactly one of {MoE, heterogeneous tiers, serving, online}:

| Neighbor | ID | Has | Gives up (vs TierShift) |
|---|---|---|---|
| [[Aurora]] | 2410.17043 | optimal MoE expert placement + comm sched on **hetero** GPUs (proves colocate-on-hetero NP-hard) | **static/offline** — assumes the traffic matrix is known a priori; names "online dynamic rebalancing" as the gap |
| [[MegaScale-Infer]] | 2504.02263 | MoE serving across **hetero tiers** (H20 attention + L40S experts), production scale | **static placement + greedy *historical* replication**; no online migration |
| [[HeterMoE]] | 2504.03871 | attention/expert split across GPU generations | **training, not serving**; static |
| [[Helix]] | 2406.01566 | hetero serving via max-flow + MILP | **dense LLMs** (no MoE/experts); placement run once |
| [[HexGen-2]] | 2502.07903 | **PD-disaggregated** hetero serving | **dense LLMs**; offline placement-style |
| PROBE / CRAFT / Patterns-behind-Chaos | 2602.00509 / 2603.28768 / 2510.05497 | **online** exploit of expert non-stationarity (prefetch / replicate / forecast) | **homogeneous GPUs** — "where" is trivial (all GPUs equal); no tier-aware cost, no cross-tier migration |
| [[Toward-Efficient-MoE-Inference]] | NeurIPS'24 | **Expert Buffering** — reactive cache of hot experts, exploits temporal locality (the Layer-0 insight we build on) | **memory caching** GPU↔CPU (stall-on-miss); not compute-tier load-balancing, no token-split, no GPU-tier gap |
| [[SwapMoE]] | 2308.15030 | dynamic resident "Virtual Experts" set, tunable memory | **approximate** (accepts accuracy drop), **single-device** memory tiering, no hardware heterogeneity |
| [[HexAGenT]] | 2605.16637 | hetero PD **routing**, agentic | dense; routes over fixed pools, no expert placement |

**No published work occupies the cell:** *MoE × heterogeneous tiers × serving × **online** expert placement/migration under a tier-cost model.* The static-placement owners (Aurora, MegaScale) punt on *online*; the online crowd (PROBE, CRAFT, Patterns) is *homogeneous* so their placement axis is degenerate. **That** is the defensible novelty — not "exploit expert non-stationarity" (the homogeneous crowd does that), and not "place experts on hetero GPUs" (Aurora does that, statically), but the *intersection*.

**The objective is a latency/cost target, not raw throughput.** The whole reason to run a *mixed* fleet (many cheap slow GPUs + a few fast ones) instead of all-fast-GPUs is **cost** — so the win is stated as *meeting a **p99 latency SLO** at lower **$/token** on a heterogeneous fleet.* A hot expert stranded on the slow tier is a straggler → a p99 tail spike → an SLO violation; tier-caching the hot experts is what holds the SLO on a cheap fleet. We measure **p99 latency / SLO-attainment, $/token (fleet-cost), throughput, and per-tier idle/straggler**; we do **not** claim model quality (serving a fixed checkpoint — quality is the model's, not ours). Precedent: Aurora/MegaScale-Infer/Mélange headline cost/SLO with no quality claim.

## 2. B2 — Characterize first (and why)

Before building a controller, measure **how expert hotness actually moves** on a real serving stream, on real heterogeneous tiers, and show a **static expert→tier placement is mis-sized for a moving target**. **At this stage the charts *are* the contribution** — they either show expert-hotness non-stationarity interacts with tier heterogeneity strongly enough to defeat static placement (justifying a controller) or they don't. Critically, B2 must also answer the **kill-question**: *does hotness shift slower than an expert can migrate?* If experts churn faster than PCIe can move them, online migration is dead on arrival and the contribution collapses to smarter static placement.

Six questions, six charts:

| #   | Question | What it determines |
| --- | --- | --- |
| Q1  | How skewed and **non-stationary** is per-expert load? (per-layer token-load distribution over wall-clock + across request domains) | **The core motivation** — does the hot set move enough to defeat a static placement? |
| Q2  | Under a static expert→tier placement, how much **slow-tier straggler** time / **fast-tier underuse** does hotness drift cause? (per-tier busy% + straggler wait vs. the fixed placement) | Sizes the win; connects hotness drift → tier imbalance |
| Q3  | **Migration cost vs. hotness timescale** — expert weight size, tier↔tier PCIe transfer time (no NVLink), vs. how fast the hot set turns over | **The kill-question** — is migration faster than the thing it chases? If no, pivot to replication-only |
| Q4  | **Replication payoff** — replicate hottest experts on the fast tier vs. migrate; VRAM cost of K replicas | Which mechanism (migrate vs replicate) wins, and the memory budget |
| Q5  | **Predictability** — can near-term per-expert hotness be forecast from recent routing? (simple predictor accuracy, lead time) | Whether the controller can act *ahead* of the shift (cf. "Prediction Is All MoE Needs") |
| Q6  | **Workload-mix sensitivity** — does the hot set shift with request **domain** (code / chat / math)? | Whether a **domain-aware** predictor is feasible (else routing-only control) |
| Q7  | (suppl.) cross-tier all-to-all bytes + per-role HBM-bw + queue depths | All-to-all/PCIe pressure as controller inputs |

Reported as **p50/p95/p99** over steady-state windows. **Success:** if Q1/Q2 show the hot set is non-stationary enough that any single static placement leaves measurable straggler/idle, **and Q3 shows migration is faster than the churn**, B2 stands alone (characterization) *and* is the load-bearing motivation for the controller. *(If Q3 fails — churn faster than migration — the finding flips to "replication-only, no migration," still publishable as the negative result that bounds the design space.)*

## 3. Main idea going forward (the system)

**TierShift: an online controller that places, migrates, and replicates MoE experts across heterogeneous GPU tiers** to keep the fast tier saturated with hot experts and eliminate slow-tier stragglers — replacing static/offline placement (Aurora) and homogeneous-blind replication (CRAFT/PROBE). It **reacts to measured per-expert load** (no workload forecast) and reallocates experts across tiers under a migration-cost + VRAM budget, with the objective of **holding a p99 latency SLO at minimum $/token on the mixed fleet** — i.e. serve most experts on cheap slow GPUs, keep the hot ones fast enough to meet the SLO.

**It's a reactive cache, not a predictor.** TierShift does **not forecast the workload** (fragile, workload-specific). Like a CPU cache or [[Toward-Efficient-MoE-Inference]]'s **Expert Buffering**, it *reacts* to measured expert load — a signal the router gives for free — and keeps recently-hot experts on the fast tier. No workload model → it self-adapts to *anyone's* traffic, which is what makes it usable as a reference (see "Layered novelty" below).

Composable mechanisms:

| # | Mechanism | What it does |
|---|---|---|
| M1 | **Reactive tier-cache** | Keep recently-hot experts resident on the **fast** tier, demote cold ones to the **slow** tier, with hysteresis to avoid thrash. Borrows the *temporal-locality* insight from [[Toward-Efficient-MoE-Inference]] / [[SwapMoE]] — but as **load-balancing across two compute tiers**, not memory caching (the slow tier still *computes*) |
| M2 | **Replicate-and-split** | For the hottest experts, hold copies on **both** tiers and split their tokens by a continuous ratio under a VRAM budget. The lever a memory-cache has **no analog for** — and the one homogeneous replication (CRAFT) can't express because its tiers are identical |
| M3 | **Tier-gap-aware policy** | Decide *migrate vs replicate vs leave* per expert from measured load × tier-gap × migration-cost (PCIe budget). **This is what must beat the naive LRU-port** (Layer 1 below) — else there's no contribution |
| M4 (optional) | **Prefetch** | Move a rising expert a beat early *only* to hide migration latency. Unnecessary when migration (~1 ms) ≪ window (~1 s); kept dormant, activated only if B2 shows it's needed |

**Layered novelty (the honest decomposition — what's borrowed vs. ours):**
- **Layer 0 (borrowed, cited):** expert hotness has temporal locality → a reactive policy works. From [[Toward-Efficient-MoE-Inference]] (Expert Buffering). We stand on it; we don't re-claim it.
- **Layer 1 (the copy = a BASELINE, not our contribution):** naive LRU/LFU "hot experts → fast GPU." This is "just porting Expert Buffering to two GPUs." We ship it *as a baseline*.
- **Layer 2 (the contribution):** because the slow tier *computes* (load-balancing, not caching) and experts can be **replicated + token-split**, the optimal policy is **not** the naive cache. **The paper exists iff Layer 2 measurably beats Layer 1.**
- **Layer 3 (the science / the reference value):** the **regime map** — when migrate vs replicate vs static wins, as a function of (locality/dwell × tier-gap × migration-cost). The transferable artifact others apply to *their* setup.

**Why this isn't the retired ideas:** not generic expert load-balancing (homogeneous EPLB/CRAFT — equal devices, no tier asymmetry); not static hetero placement (Aurora owns it, proved the hard case); not attention/expert tier-splitting (HeterMoE/MegaScale own it); **and not just Expert Buffering** (memory-capacity caching with a stall-on-miss model, GPU↔CPU — TierShift is straggler-minimizing load-balancing across two *compute* tiers, GPU↔GPU, with token-splitting).

**Headline target:** On a heterogeneous MoE fleet, TierShift **holds a target p99 latency SLO at lower $/token than an all-fast-GPU deployment** — serving most experts on cheap slow-tier GPUs while reactively keeping the hot ones on the fast tier. Under workload drift where static placement **violates the SLO** (the hot set migrates onto the slow tier), TierShift holds it — *without* hand-tuning or a workload model — and **measurably beats the naive LRU-port (Layer 1)** via replicate-and-split.

**Evaluation:** two framings of the same runs — *(i) fixed fleet → SLO-attainment + throughput; (ii) fixed SLO → $/token (min-cost fleet that still meets p99).* Baselines = (a) static-best placement (Aurora-style offline optimum for the average mix), (b) **naive cache-port** (LRU hot→fast, the Layer-1 "just copy Expert Buffering" baseline — *the one we must beat*), (c) homogeneous replication ported blind to tiers (CRAFT/PROBE-style), (d) round-robin EP, (e) **all-fast-GPU** (the expensive SLO-meeting reference for the cost story). Ablate baseline → +M1 (reactive cache) → +M2 (replicate-split) → +M3 (tier-gap policy). **Success criteria:** (1) **$/token at fixed p99 SLO** below the all-fast reference; (2) **p99 SLO held under drift** where static (a) violates it; (3) **make-or-break:** headroom of (M1+M2+M3) over baseline (b). Supporting metrics: throughput (tok/s), per-tier GPU busy%, slow-tier straggler wait, cross-tier migration bytes/s, imbalance ratio over time.

## 4. How we'll use WatGPU (initial phase)

Same node as [[Research Plan]] — **watgpu308**, 8 GPUs = 4× L40S + 2× RTX A6000 + 2× RTX 6000 Ada — *a heterogeneous cluster in a box*: a **fast tier** (Ada / L40S, FP8-capable) and a **slow tier** (Ampere A6000, BF16), a real ~2.4× compute gap. For TierShift this maps cleanly onto **fast-tier expert slots vs slow-tier expert slots**, with experts migrating between them over PCIe (no NVLink → the migration cost is real and measurable, which is the point).

**Initial phase — make it serve + profile hotness×tier.** Stand up **EP MoE serving across both tiers** on a small model (**OLMoE-1B-7B**: 64 experts / 8 active — enough experts to exhibit skew), replay a mixed request trace (chat + code + math), and emit Q1–Q7 — the hotness-×-tier profile that motivates the controller. *(Caveat, mirroring Plan 1: OLMoE fits one card, so cross-tier EP is partly artificial at this scale — treat the small-model numbers as harness-proving anchors; real tier pressure shows at the 30B phase.)*

Then **swap to Qwen3-30B-A3B** (128 experts, FP8, genuinely needs multi-GPU EP across tiers) reusing the same harness. Deferred: 16-GPU / multi-node, the controller's online migration at scale, and a cloud run for the controlled eval where a stable 2-tier shape and statistical power matter.

---

## Workload

| Dim | Spec |
| --- | --- |
| Model | Qwen3-30B-A3B (serving target, 128 experts); OLMoE-1B-7B for bring-up (64 experts) |
| Task | **Inference serving only** — no fine-tuning, no RL, no training loop |
| Request stream | trace replay: ShareGPT + LMSYS-Chat-1M + a code/math slice, **mixed and time-varying** (to induce hot-set drift) |
| Stack | vLLM or SGLang in **expert-parallel** mode, experts pinned per-tier; custom telemetry + migration hooks in the fork |
| Heterogeneity | fast tier (L40S / RTX 6000 Ada) vs slow tier (RTX A6000); PCIe Gen4, no NVLink |
| Control | **reactive** tier-cache (no workload prediction) + tier-aware replicate/split under VRAM + PCIe budget |

## Out of scope
Model quality / accuracy claims (we serve a fixed checkpoint) · training or RL of any kind · new MoE architecture or router · dense-LLM hetero serving (Helix/HexGen-2 own it) · attention/expert tier-splitting (HeterMoE/MegaScale own it) · homogeneous EP load-balancing (CRAFT/EPLB own it) · multi-node / 16-GPU scale this phase.

## Top risks → fallbacks
- **Kill-risk (Q3): hotness churns faster than PCIe migration** → fall back to *replication-only* (still novel as tier-aware replication vs. CRAFT's homogeneous replication); report the timescale as the finding that bounds the design.
- Small-model EP is artificial on 1 card → anchor on OLMoE, prove the real claim on Qwen3-30B-A3B.
- vLLM/SGLang EP doesn't expose per-expert load / hot-swap → fork to emit per-expert token counts + add a migrate hook (the Track-A-style fork work).
- Cross-tier all-to-all over PCIe dominates → measure it (Q7); it is a *real hetero cost*, not noise.
- "Crowded neighborhood" (2602–2606 arXiv churn) → the wedge is the **tier × online** intersection; keep the eval pinned to "migration-across-tiers beats replication-on-homogeneous," the one thing neighbors can't show.

## Related notes
`papers/` boundary set: [[Aurora]] · [[MegaScale-Infer]] · [[HeterMoE]] · [[Helix]] · [[HexGen-2]] · [[HexAGenT]] · dynamic-homogeneous cluster (PROBE 2602.00509, CRAFT 2603.28768, Patterns-behind-Chaos 2510.05497). Sibling plan: [[Research Plan]] (AutoPD-RL). First-milestone comparison: [[Benchmark Comparison]]. Cluster: [[WatGPU]].
