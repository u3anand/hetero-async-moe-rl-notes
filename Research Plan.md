# Research Plan

**Throughput-optimal asynchronous MoE RL on heterogeneous GPU clusters for agentic workloads.**

A workload-characterization benchmark (B1) as the precondition, then a heterogeneity-aware scheduler with composable mechanisms as the system contribution (B2).

---

## 1. Motivation

RL post-training of **Mixture-of-Experts** models is how frontier coding/agentic systems are now built (GLM-4.x, Qwen3-Coder, INTELLECT-3, etc.). But the open systems literature only characterizes and schedules this kind of RL for **dense** models on **math** workloads. Three things break when you move to the regime that actually matters — **MoE × asynchronous × multi-replica × heterogeneous compute × agentic**:

1. **Agentic episodes are not math episodes.** SWE-style agent rollouts have heavy-tailed wallclock, **bimodal** tool-call latency (fast file ops vs slow test/build runs), and real sandbox I/O. Short, homogeneous, GPU-bound math episodes hide all of this.
2. **MoE changes where time goes.** Expert-parallel all-to-all and weight broadcast (~95% expert bytes / ~5% router) reshape the cost balance. AReaL-Hex's assumption that rollout cost dominates training cost (`C_I > C_T`) may not hold on MoE — and if it inverts, the scheduling problem changes.
3. **Heterogeneous tiers differ in compute *and* precision.** A fast tier (Ada, FP8-capable) and a slow tier (Ampere, BF16-only) don't just run at different speeds — they create **asymmetric expert staleness** and **cross-replica router drift** that no dense scheduler models.

No published work occupies this five-dimensional cell, characterized for throughput and scheduled as a system. Each neighbor gives up at least one dimension:

| Neighbor | Has | Gives up |
|---|---|---|
| AReaL-Hex | hetero × async × MILP scheduler × throughput | dense, math, no MoE; assumes `C_I > C_T` |
| ReLibra | MoE × hetero-*bandwidth* × RL × system | compute-homogeneous, single-replica |
| R3 / Router-Aware IS | MoE × within-replica × stability | no system, no hetero, no multi-replica |
| HeterMoE / Lazarus / LAER-MoE | MoE × hetero | pretraining, not RL; no rollout/drift |
| StreamRL / RollArt / ROLLMUX | hetero × async × RL | dense, math |
| SWE-World / SWE-MiniSandbox | agentic × sandbox optimization | not MoE, not hetero, not scheduling |

We measure system throughput in steady-state windows; we do **not** claim final RL quality (full convergence on a 30B MoE is infeasible at our scale). Precedent: AReaL-Hex headlines "1.31–1.50× throughput, 1.46× cost" with no convergence claim.

## 2. B1 — Characterize first (and why)

Before designing a scheduler, measure where time actually goes on this workload, on real heterogeneous hardware. **At this stage the charts *are* the contribution** — they either show the workload is structurally different from dense math RL (justifying a new scheduler) or they don't. They also produce the cost inputs the scheduler in §3 consumes.

Six questions, six charts (+ one supplementary):

| #   | Question                                                                  | What it determines                                    |
| --- | ------------------------------------------------------------------------- | ----------------------------------------------------- |
| Q1  | Where does per-episode wallclock go? (per-tool-call wallclock × category) | Is tool I/O bimodal; where to invest mechanism effort |
| Q2  | Is the GPU starved? (GPU busy % + sandbox queue depth)                    | Whether tool-aware scheduling (M4) is needed          |
| Q3  | Does `C_I > C_T` hold for MoE? (isolate inference vs train cost)          | The central scheduling assumption — confirm or refute |
| Q4  | Weight-broadcast cost and split (router bytes/time vs expert)             | Sizes the decoupled-broadcast win (M1)                |
| Q5  | How heavy-tailed are episodes? (per-episode wallclock CDF)                | Concurrency formula + scheduler cost terms            |
| Q6  | Cross-replica router-KL drift vs tier-speed gap                           | Sizes the cross-replica routing-replay win (M3)       |
| Q7  | (suppl.) train MFU + rollout HBM-bandwidth utilization                    | Kernel-level perf sanity check                        |

Reported as **p50/p95/p99** over steady-state windows (heavy-tailed → means hide the story). **Success:** if ≥5 of 6 diverge from prior dense/math characterizations, B1 stands alone (workshop / characterization track); otherwise it is the load-bearing motivation section of the system paper.

## 3. Main idea going forward (the system)

A **heterogeneity-aware scheduler** that extends AReaL-Hex's MILP with expert-placement variables, a router-KL constraint, and tier-bandwidth-aware all-to-all cost — paired with lightweight **runtime controllers** that maintain the MILP setpoints. Static placement (MILP solved once) + dynamic feedback control; no re-solve mid-run.

Three composable mechanisms the scheduler controls:

| # | Mechanism | What it does |
|---|---|---|
| M1 | Decoupled router/expert broadcast | Router (~5% of params) high cadence; experts (~95%) low cadence |
| M2 | Tier-aware EP all-to-all | Expert placement + routing avoid lock-step to the slow tier (likely the largest single win) |
| M3 | Cross-replica routing replay | Extends R3 with per-token router-version tags; train side reconciles cross-replica drift |

**M4 (conditional) — tool-execution-aware co-scheduling:** sandbox CPU as a first-class MILP resource. Kept as a *dormant* variable in the formulation from the start; activated only if B1 Q2 shows GPU starvation from sandbox contention (activation is a flag, not a redesign).

MILP extensions over AReaL-Hex's `(D_T, D_I, σ, τ)`: new variables `e_{i,t}` (per-expert tier placement), `β_router/β_expert` (per-group broadcast cadence), explicit replica count `K`, and dormant sandbox `k_s/R_s`; new constraints `KL(R_replica || R_trainer) ≤ ε` and tier-bandwidth-respecting all-to-all cost; same objective shape `min max(C_T, C_I)`.

**Headline target:** the full system on a naive-hetero cluster recovers ≥80% of homogeneous-fastest throughput at heterogeneous cost (AReaL-Hex reported 1.31–1.50× at fixed budget).

**Evaluation:** for three baseline configs (homog-fastest, homog-slowest, naive-hetero), ablate baseline → +M1 → +M3 → +M2 → full system (+M4 only if Q2 triggers).

## 4. How we'll use WatGPU (initial phase)

Reliable hardware is a single node — **watgpu308**, 8 GPUs = 4× L40S + 2× RTX A6000 + 2× RTX 6000 Ada. That one node is effectively *a heterogeneous cluster in a box*: a **fast tier** (Ada, FP8-capable: L40S + RTX 6000 Ada) and a genuine **slow tier** (Ampere RTX A6000, BF16-only) — a real ~2.4× compute gap plus an FP8-vs-no-FP8 split. Cluster constraints (preemption, 32 cores, no NVLink, conda/udocker workflow) are in [[WatGPU]].

**Initial phase — make it work + the 3-config comparison.** Bring the full async loop up on a small MoE (**OLMoE-1B-7B**, fits one card → no TP/FP8) and run the three baseline configs (homo-fast / homo-slow / hetero) on watgpu308, emitting Q1–Q7. Full step-by-step plan with GPU pools and gates: **[[Initial Plan]]**.

Then **swap to Qwen3-30B-A3B** (FP8 + TP=2) reusing the same harness unchanged. Deferred: 16-GPU / multi-node, the full tier-mix sweep, and AWS for the controlled eval where statistical power and a stable 2-tier shape matter.

---

## Workload

| Dim              | Spec                                                                                    |
| ---------------- | --------------------------------------------------------------------------------------- |
| Model            | Qwen3-30B-A3B (30B total, ~3B active, 128 experts); OLMoE-1B-7B for bring-up            |
| Fine-tuning      | LoRA (full FT optional)                                                                 |
| RL algorithm     | GRPO with R3 (rollout routing replay)                                                   |
| Agent            | mini-swe-agent — bash-only tool surface                                                 |
| Sandbox / reward | per-episode container (rootless udocker on WatGPU); reward = the repo's real test suite |
| Dataset          | `nebius/SWE-rebench-openhands-trajectories` (67K trajectories, 1,823 Python repos)      |
| Stack            | prime-rl orchestration · PyTorch FSDP2 trainer · vLLM (FP8) inference · DTensor EP      |
| Concurrency      | asynchronous multi-replica rollout; bounded staleness `η`                               |

## Out of scope
New RL algorithm (GRPO unchanged) · new optimizer or parallelism axis · final SWE-bench pass-rate / convergence-quality claims · dynamic/elastic cluster membership · repo-prefix KV cache · multimodal/VLA RL · Megatron port of the mechanisms (slime port is a stretch robustness experiment).

## Related notes
[[WatGPU]] (cluster) · [[Initial Plan]] (initial phase) · concept notes: [[MoE vs Dense Workload]], [[MoE Architecture]], [[Inference]], [[Training]], [[Transformer]] · paper notes in `Papers/` ([[Papers/AReaL-Hex]], [[Papers/ReLibra]], [[Papers/R3]], [[Papers/Router-Aware IS]], [[Papers/prime-rl]], [[Papers/slime]], [[Papers/HeterMoE]], [[Papers/Lazarus]]).
