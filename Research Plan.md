# Research Plan

**AutoPD-RL — automatic prefill/decode allocation for agentic RL rollouts on heterogeneous GPUs.**

A workload-characterization phase (B1) that profiles where prefill/decode/env demand goes and shows static PD configuration fails under non-stationary multi-turn rollout — then an online PD controller that automatically sizes/routes prefill vs decode pools as the system contribution (B2).

> Pivoted 2026-06-04 from the prior MoE-hetero-MILP thesis. Validation of the new direction (verified against primary sources): [[Direction Validation]]. Curated literature: `papers/` ([[RollArt]], [[HexAGenT]], [[Heddle]], [[TokenScale]], [[DynaServe]], …).

---

## 1. Motivation

Agentic RL post-training (SWE agents, tool use) runs a **rollout** loop — multi-turn generation interleaved with tool/environment execution — that feeds a **trainer**. On heterogeneous GPU clusters the rollout is best served **disaggregated**: compute-bound **prefill** (ingesting growing context + tool outputs) and bandwidth-bound **decode** (token generation) routed to best-fit hardware. [[RollArt]] establishes exactly this for agentic RL and shows the win — but configures the prefill/decode instance split **manually and statically**, and its authors **explicitly name automatic configuration as future work** (verbatim: *"real deployments currently require manual configuration of prefill and decoding instances, which easily leads to load imbalance. We hence leave it as future work."*).

That gap is real and load-bearing because **agentic RL rollout creates time-varying PD demand**: multi-turn interaction, growing context, variable tool-output lengths, environment delays, and a heavy-tailed mix of episodes all shift the prefill:decode pressure ratio over the course of training. A static split → load imbalance, idle GPUs, and — critically — the trainer starving. ([[RollArt]]'s own production bottleneck is the blocking `get_batch` on the **SampleBuffer**, *up to 62% of iteration time as GPU idleness*, waiting for enough complete trajectories.)

**Why the serving world hasn't already solved this.** Dynamic prefill/decode autoscaling is, by itself, *not* novel — serving systems already do it: [[TokenScale]] (token-velocity scaling + convertible decoders), [[DOPD]], [[Arrow]], [[DynaServe]] all adjust the PD ratio online; [[DistServe]]/[[Splitwise]] are the static ancestors. But every one assumes **static weights + an SLO-driven request stream**. RL rollout breaks all three of their assumptions:

1. **Weights update every step** — warm-start / KV-reuse / instance-stability assumptions break, and PD layout interacts with cross-cluster weight sync.
2. **Batch-synchronous long tail** — >90% of rollout runtime sits in the tail; the objective is *batch completion*, not per-request latency.
3. **The downstream objective is the trainer, not a user** — what matters is **fresh complete trajectories per training deadline** (cf. [[RollArt]] SampleBuffer idle; [[Freshness-Aware-PER]] formalizes freshness on the algorithmic side), not request latency.

So serving PD-autoscalers don't port unmodified — and **that** is the defensible novelty, not the autoscaling mechanism.

**No published work occupies this cell.** Each neighbor gives up at least one dimension:

| Neighbor | ID | Has | Gives up (vs AutoPD-RL) |
|---|---|---|---|
| [[RollArt]] | 2512.22560 | hetero PD for agentic RL, hardware-affinity | **static/manual PD config** (names it as future work) |
| [[HexAGenT]] | 2605.16637 | PD routing on hetero GPUs, agentic workflows | **fixed pools (routing not sizing); serving SLO objective** |
| [[Heddle]] | 2603.28101 | trajectory scheduling, runtime prediction, long-tail | per-trajectory **MP tuning only**; calls PD-sizing *orthogonal* |
| [[ROSE]] | 2605.06534 | elastic rollout on spare serving GPUs | lever is harvest-idle-GPU, **not PD sizing** |
| [[TokenScale]]/[[DynaServe]]/[[DOPD]]/[[Arrow]] | serving | **dynamic PD-ratio control** | static weights + SLO; **not RL rollout** |
| [[RollPacker]] | 2509.21009 | long-tail mitigation (synchronous RL) | batch-round batching, **not PD resource sizing** |
| [[EARL]] | 2510.05943 | dynamic parallelism under long context | parallelism layout, **not PD pool ratio** |

We measure **rollout throughput and trainer idle time** in steady-state windows; we do **not** claim final RL quality (full convergence at our scale is infeasible). Precedent: [[RollArt]]/AReaL-style systems headline throughput/cost with no convergence claim.

## 2. B1 — Characterize first (and why)

Before building a controller, measure **where prefill/decode/env demand actually goes** on agentic RL rollout, on real heterogeneous hardware, and show a **static PD split is mis-sized for a moving target**. **At this stage the charts *are* the contribution** — they either show PD demand is non-stationary enough to defeat static configuration (justifying a controller) or they don't. They also produce the demand signals the controller in §3 consumes.

Six questions, six charts (+ one supplementary):

| #   | Question | What it determines |
| --- | --- | --- |
| Q1  | Where does per-turn time go? (prefill vs decode vs tool/env, by tool category) | Whether PD demand is the right knob; where env-heavy phases sit |
| Q2  | How non-stationary is the prefill:decode ratio? (P:D demand over rollout time, per phase) | **The core motivation** — does a static split mis-fit a moving target? |
| Q3  | Does a static PD pool cause idle GPUs *and* trainer starvation? (GPU busy% by worker type + SampleBuffer fill + trainer idle) | Sizes the win; connects PD imbalance → freshness/idle |
| Q4  | Context growth + TTFT/TPOT trajectories (per-turn input/output tokens, TTFT, TPOT) | The predictable signals a controller can act on |
| Q5  | How heavy-tailed are episodes, and do they switch PD regime? (per-episode wallclock CDF + per-episode P/D/env phase mix) | Whether episodes shift prefill-heavy↔decode-heavy↔env-heavy |
| Q6  | Do SWE signals predict PD/env class? (repo/test/build/cache/tool features → prefill-heavy / decode-heavy / env-heavy / failure-prone) | Whether a **SWE-aware predictor** is feasible (else demand-only control) |
| Q7  | (suppl.) rollout HBM-bw + prefill/decode queue depths | Saturation sanity + queue-pressure controller inputs |

Reported as **p50/p95/p99** over steady-state windows (heavy-tailed → means hide the story). **Success:** if Q2/Q3/Q5 show PD demand is non-stationary enough that any single static split leaves measurable idle/starvation, B1 stands alone (characterization track) *and* is the load-bearing motivation for B2.

## 3. Main idea going forward (the system)

**AutoPD-RL: an online controller that automatically sizes and routes the prefill/decode pools for agentic RL rollout on heterogeneous GPUs**, replacing RollArt's manual `hw_mapping`/static instance counts. Predict prefill/decode pressure from rollout signals; dynamically reallocate or route rollout requests across heterogeneous inference workers to keep both pools busy **and** maximize fresh completed trajectories per training deadline.

Composable mechanisms:

| # | Mechanism | What it does |
|---|---|---|
| M1 | **Demand predictor** | From per-turn token counts, context growth, TTFT/TPOT, queue depths → forecast near-term prefill vs decode pressure (borrows the runtime-prediction idea from [[Heddle]], retargeted to PD pressure not trajectory length) |
| M2 | **PD pool sizing / role-flip** | Adjust prefill:decode instance counts online; convertible workers flip role under bursts (adapts [[TokenScale]]/[[DynaServe]] mechanisms — then shows why they need RL-specific modification) |
| M3 | **Freshness-aware routing** | Route/prioritize so the trainer's `get_batch` deadline is met (objective = fresh complete trajectories, not request latency); SampleBuffer fill + trajectory age as control inputs |

**M4 (conditional) — SWE-aware predictor:** if B1 Q6 shows repo/test/build/cache/tool signals predict a trajectory's PD/env class, fold those features into M1. Kept dormant; activated only if Q6 is positive (a flag, not a redesign).

**Why this isn't the retired ideas:** it is **not** generic trajectory scheduling ([[Heddle]]'s territory — and Heddle calls PD-sizing orthogonal); it is **not** generic PD disaggregation (serving systems own that); it is **not** a RollArt replacement (RollArt provides the disaggregated substrate — AutoPD-RL is the *automatic* layer it leaves open). The retired SampleBuffer-freshness idea returns as the **objective (M3)**, not as a scheduling mechanism.

**Headline target:** AutoPD-RL on a naive/static-PD heterogeneous cluster recovers most of the throughput a hand-tuned PD split achieves — *without* the hand-tuning — and reduces trainer idle vs the static baseline across changing workload mixes.

**Evaluation:** baselines = static-PD configs (RollArt-style manual splits: prefill-heavy, decode-heavy, balanced) + a serving-autoscaler-ported-naively baseline; ablate baseline → +M1 → +M2 → +M3 → full (+M4 if Q6 triggers). Metrics: rollout throughput, trainer idle %, fresh-trajectories-per-deadline, GPU busy% by worker type.

## 4. How we'll use WatGPU (initial phase)

Reliable hardware is a single node — **watgpu308**, 8 GPUs = 4× L40S + 2× RTX A6000 + 2× RTX 6000 Ada — effectively *a heterogeneous cluster in a box*: a **fast tier** (Ada, FP8-capable: L40S + RTX 6000 Ada) and a **slow tier** (Ampere RTX A6000, BF16-only), a real ~2.4× compute gap. For AutoPD-RL this maps cleanly onto **prefill workers vs decode workers on different tiers**. Cluster constraints (preemption, 32 cores, no NVLink, udocker workflow) in [[WatGPU]].

**Initial phase — make it work + profile PD demand.** Bring the full async agentic-RL rollout loop up on a small model (**OLMoE-1B-7B**, fits one card → no TP/FP8), disaggregate prefill/decode across tiers, run the static-PD baseline configs, and emit Q1–Q7 — the PD-demand profile that motivates the controller. Full step-by-step plan with GPU pools and gates: **[[Initial Plan]]**.

Then **swap to Qwen3-30B-A3B** (FP8 + TP=2) reusing the same harness unchanged. Deferred: 16-GPU / multi-node, the controller's online re-sizing at scale, and AWS for the controlled eval where statistical power and a stable 2-tier shape matter.

---

## Workload

| Dim | Spec |
| --- | --- |
| Model | Qwen3-30B-A3B (agentic RL target); OLMoE-1B-7B for bring-up (MoE incidental now, not the thesis) |
| Fine-tuning | LoRA (full FT optional) |
| RL algorithm | GRPO (unchanged — not a contribution) |
| Agent | mini-swe-agent — bash-only tool surface |
| Sandbox / reward | per-episode container (rootless udocker on WatGPU); reward = the repo's real test suite |
| Dataset | `nebius/SWE-rebench-openhands-trajectories` (67K trajectories, 1,823 Python repos) |
| Stack | prime-rl orchestration · PyTorch FSDP2 trainer · vLLM (FP8) inference, **prefill/decode-disaggregated** · udocker env |
| Concurrency | asynchronous multi-replica rollout; bounded staleness `η`; trainer SampleBuffer |

## Out of scope
New RL algorithm (GRPO unchanged) · new optimizer/parallelism axis · final SWE-bench pass-rate / convergence claims · MoE-internals scheduling (expert placement, router drift — retired with the old thesis) · multimodal/VLA RL · the serving-SLO objective (we optimize trainer-facing freshness/throughput, not request latency).

## Related notes
[[Direction Validation]] (the verified wedge) · [[WatGPU]] (cluster) · [[Initial Plan]] (initial phase) · concept notes: [[Inference]], [[Training]], [[Transformer]] · paper notes in `papers/` ([[RollArt]], [[HexAGenT]], [[Heddle]], [[ROSE]], [[TokenScale]], [[DynaServe]], [[DistServe]], [[Splitwise]], [[RollPacker]], [[EARL]], [[ROLL]], [[Freshness-Aware-PER]], [[RLix]]).
