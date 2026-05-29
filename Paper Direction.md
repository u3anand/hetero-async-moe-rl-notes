# Paper Direction

Updated plan as of 2026-05-29. Supersedes prior revisions of this doc and the open-ended search in [[research-direction]] memory.

## One-line thesis

**Throughput-optimal asynchronous MoE RL on heterogeneous GPU clusters for agentic workloads, with a workload-characterization benchmark as the precondition and a hetero-aware scheduler with composable mechanisms as the system contribution.**

## Structure: two coupled contributions

- **B1 — Benchmark.** RL-specific throughput characterization for async multi-replica MoE RL on a real agentic workload, on hetero hardware. Establishes the metric scaffolding and shows where the bottlenecks actually live.
- **B2 — System.** A hetero-aware scheduler (extending [[Papers/AReaL-Hex|AReaL-Hex]]'s MILP formulation) controlling three composable mechanisms targeting the bottlenecks B1 surfaces.

B1 is the precondition for B2. Either could be its own paper (B1 → MLSys workshop or characterization track, B2 → MLSys/EuroSys main track), but the natural shape is B1 as §3 of B2 ("Workload Characterization") and the system as §§4–6.

## Why this workload (justification)

The workload is **Qwen3-30B-A3B on agentic SWE-RL via mini-swe-agent + Docker sandbox + GRPO + R3 routing replay**, on top of [[Papers/prime-rl|prime-rl]] + FSDP2 + vLLM. This is defensible because:

1. **Structural mismatch with prior work.** [[Papers/AReaL-Hex|AReaL-Hex]] and other hetero RL papers evaluate on mathematical reasoning — short, homogeneous episodes, GPU-bound, no tool I/O. Agentic SWE-RL has heavy-tailed episode durations, bimodal bash-call latency distribution (fast file ops + slow test runs), and significant sandbox I/O. These properties fundamentally change the scheduling problem.
2. **Industry relevance.** Production agentic MoEs (GLM-4.6/4.7, Qwen3-Coder-Next, INTELLECT-3.x, Claude coding) all train on this workload shape. The community already publishes engineering recipes for it (prime-rl, slime, Miles) without throughput characterization.
3. **MoE-relevance.** Qwen3-30B-A3B is a production MoE, not a toy. SWE-RL is one of the workloads MoEs are actively deployed on.
4. **Public data exists.** SWE-bench Verified, SWE-rebench, SWE-Gym — citable, reproducible. The `nebius/SWE-rebench-openhands-trajectories` corpus is what prime-rl's recipe consumes.
5. **Exposes phenomena prior work hides.** Meta's SWE-RL paper explicitly chose *rule-based* reward to avoid tool-execution costs. AReaL-Hex's MILP assumes `C_rollout > C_train` and uniform episode duration. Both assumptions break on this workload. Demonstrating the break is itself a contribution.

## Critical framing: throughput, not quality

The paper is a **systems paper**. It measures **system throughput** in steady-state windows, not final RL quality. Full GRPO convergence on Qwen3-30B-A3B + agentic SWE-RL takes weeks-to-months at production scale; reproducing it in our experiments is neither feasible nor the contribution.

Precedent: [[Papers/AReaL-Hex|AReaL-Hex]]'s headline is "1.31–1.50× throughput, 1.46× cost reduction" — they make no convergence quality claims. MLSys / EuroSys accept this; SOSP would push harder.

Claims we make:
- End-to-end system throughput (env-steps/sec)
- Useful-token throughput (off-policy-corrected)
- Hetero efficiency (achieved / homog-fastest-tier)
- Router KL trajectory and loss curve sanity checks

Claims we **do not** make:
- Final SWE-bench pass rate
- Final RL quality vs baseline
- Convergence wallclock as a direct measurement

We argue throughput improvements transfer to convergence wallclock at any scale as a corollary, but do not demonstrate it.

## B1 — Benchmark

### Assumption

A multi-tier heterogeneous GPU cluster of ~8–24 GPUs spanning at least two compute tiers (e.g., Hopper + Ada or H100 + L40S). Source is interchangeable — academic shared infrastructure or paid AWS — and is treated as a solved access problem for the purposes of this plan. The specific tier mix affects numbers but not the methodology.

### Stack

- Framework: [[Papers/prime-rl|prime-rl]]
- Training engine: PyTorch FSDP2
- Inference engine: vLLM (FP8 supported)
- Model: Qwen3-30B-A3B (LoRA acceptable)
- Workload: agentic SWE-RL via mini-swe-agent + Docker sandbox
- Reward: real project test execution (not rule-based)
- RL algorithm: GRPO with R3 (rollout routing replay) enabled
- Parallelism: FSDP2 sharding + EP for MoE layers + CP for long context

### Metrics

Per-episode telemetry:
- Start/end timestamps
- Token counts
- Per-tool-call wallclock + category tag (file_op / git / test / build / other)
- Rollout policy version at start
- Train policy version at consumption (for staleness)
- Router KL between rollout-version and train-version

Per-update telemetry:
- Time-in-rollout, time-in-train, time-in-broadcast
- Broadcast time decomposed by param group (router vs experts)
- Useful-token count (off-policy-corrected effective gradient signal)
- EP all-to-all bytes and seconds
- Expert activation histogram

System telemetry:
- GPU busy % per worker (CUDA event timeline)
- Sandbox queue depth over time
- Tool-wait time per episode

Derived headline metrics:
- Rollout p50 / p95 / p99 latency
- Train updates/sec
- End-to-end env-steps/sec
- Useful-token throughput
- Hetero efficiency
- `C_I / C_T` ratio (rollout vs train cost, to validate or refute AReaL-Hex's regime assumption on MoE)
- Per-tool-call latency distribution

### Configurations (the sweep)

Three baseline configurations, plus sweep dimensions:

| Config | Purpose |
|---|---|
| Homog-fastest-tier | Throughput ceiling reference |
| Homog-slowest-tier | Cheap-scaling floor reference |
| Naive hetero (prime-rl as-is on mixed tiers) | The control B2 must beat |

Sweep dimensions:
- Tier mix ratio (e.g., 100/0, 75/25, 50/50, 25/75, 0/100)
- Cluster size (small, medium)
- EP degree
- Weight broadcast cadence
- Rollout concurrency per worker

Reps: 3–5 per config to characterize variance from heavy-tailed episodes.

### Expected findings to commit to

These are predictions the benchmark either confirms or refutes:

1. `C_I / C_T` ratio inverts from AReaL-Hex's dense regime on MoE (rollout becomes cheaper than train).
2. EP all-to-all dominates training wallclock on PCIe-only tier; less so on NVLink tier.
3. Weight broadcast time is non-trivial fraction of step time and is ~95% experts / ~5% router by bytes.
4. Tool-call latency distribution is bimodal: most fast (sub-second), ~5% slow (test/build, 30s–minutes).
5. Per-episode wallclock dominated by the slow 5% of tool calls.
6. Cross-replica router KL drift increases monotonically with tier-speed gap.

Each is a single chart. The set of six is the characterization paper.

## B2 — System

### Headline contribution

**A hetero-aware MILP scheduler for asynchronous MoE RL that extends [[Papers/AReaL-Hex|AReaL-Hex]]'s formulation with expert-placement variables, router-version constraints, tier-bandwidth cost terms, and (dormantly) sandbox CPU resource variables.**

### Three enabling mechanisms

The scheduler chooses among and configures these primitives:

| # | Mechanism | What it does | Throughput impact |
|---|---|---|---|
| M1 | Decoupled router/expert broadcast | Router (~5% of params) at high cadence, experts (~95%) at low cadence | Likely large; broadcast is a measurable chunk |
| M2 | Tier-aware EP all-to-all | Experts placed by tier; all-to-all does not lock-step to slowest interconnect | Probably the largest single win |
| M3 | Cross-replica routing replay with version tags | Extends R3 with per-token router-version metadata; train side reconciles cross-replica drift | Stability/quality; supports useful-token-throughput claim |

### Optional fourth mechanism (conditional on B1 findings)

**M4 — Tool-execution-aware scheduling.** Adds sandbox CPU as a first-class resource type in the MILP, co-scheduling GPU and CPU. Implemented as a dormant variable in the formulation from the start; activated as a measured mechanism only if B1 shows GPU busy % < 60% with steady-state nonzero sandbox queue depth.

If activated, M4 is potentially the most defensible novelty in the whole paper because no published work co-schedules GPU and sandbox CPU for agentic RL.

### MILP extensions (concrete)

Building on AReaL-Hex's `(D_T, D_I, σ, τ)` formulation:

New decision variables:
- `e_{i,t} ∈ {0,1}` — expert `i` placement on tier `t`
- `β_router`, `β_expert` — broadcast cadences per param group
- `η_router` — per-replica router-version slack
- `k_s` (dormant) — sandbox concurrency per rollout worker
- `R_s` (dormant) — per-repo sandbox pre-warm replication

New constraints:
- `KL(R_replica_i || R_trainer) ≤ ε` for all replicas
- Tier-bandwidth-respecting EP all-to-all cost in `C_I`
- (Dormant) sandbox CPU capacity constraint

New objective:
- Same shape (`min max(C_T, C_I)`), but `C_I = max(C_inference, C_tool) / concurrency` when M4 active

### Implementation order

Driven by what B1 surfaces as bottleneck, but default order:

1. Decoupled broadcast (M1) — easiest, likely big win
2. Cross-replica routing replay (M3) — supports correctness claim
3. Tier-aware EP (M2) — hardest, biggest single win
4. MILP scheduler — integrates everything

If B1 surfaces a different dominant bottleneck, order shifts but the set stays.

## Phases and rough timing

| Phase | Output | Approx duration |
|---|---|---|
| 0. Decisions | Framework chosen (prime-rl ✅), cluster access confirmed, maintainer outreach sent | 1 week |
| 1. B1 — Benchmark | Toolchain up, metric harness, scoping run, config sweep, characterization writeup | 4–6 weeks |
| 2. B2 design | 5-page design doc with full MILP formulation (including dormant M4 variables) | 1–2 weeks |
| 3. B2 implementation | M1, M3, M2 landed in prime-rl fork; MILP solver integrated; optional M4 | 6–8 weeks |
| 4. B2 evaluation | Ablation matrix: baseline → +M1 → +M3 → +M2 → +scheduler, on each B1 config | 2–3 weeks |
| 5. Paper writeup | Full paper draft | 2–3 weeks |

**Total: ~4 months.** Realistic for a venue submission.

## Evaluation plan

### B1 baseline configurations

As above: homog-fast, homog-slow, naive hetero. Sweep across tier mix, cluster size, parallelism degrees, broadcast cadence.

### B2 ablation matrix

For each of the three baseline configurations, run:
- Vanilla prime-rl (baseline)
- + M1 (decoupled broadcast)
- + M3 (cross-replica routing replay)
- + M2 (tier-aware EP)
- Full system (M1 + M2 + M3 + scheduler)
- Full system + M4 (only if B1 triggers it)

Primary headline: full system on naive-hetero config recovers ≥80% of homog-fastest-tier throughput at hetero cost.

Comparable target: AReaL-Hex reported 1.31–1.50× throughput at fixed budget. We target similar magnitude on MoE + agentic.

### Sanity checks (not headline)

- Loss curves overlay across configs to show training quality is preserved
- Router KL trajectories to show R3 + M3 keep routers aligned
- Per-mechanism contribution chart showing each one's marginal effect

## Risks and contingencies

1. **B1 finds a bottleneck none of M1/M2/M3 address.** Specifically: if GPU busy % < 60% with sandbox-queue contention, M4 must activate and the paper grows by one mechanism. Mitigation: M4 is in the MILP formulation from the start as a dormant variable; activation is a flag, not a redesign.
2. **prime-rl ships breaking changes during the project.** Mitigation: pin a commit at Phase 0; rebase only deliberately.
3. **ReLibra or "Missing Old Logits" papers occupy adjacent territory more than expected.** Read both before Phase 2 design. Adjust framing if needed; the four-way intersection (MoE × hetero compute × async × multi-replica × agentic) likely still differentiates.
4. **Throughput-only framing rejected at SOSP.** Mitigation: target MLSys or EuroSys; small math-RL convergence experiment as proxy evidence if needed.
5. **Tool I/O is so dominant that GPU-side mechanisms are uninteresting.** Mitigation: M4 becomes the headline instead of an optional mechanism.

## State of related work (as of 2026-05-29)

### Direct overlap (read before Phase 2 design)

- [[Papers/AReaL-Hex|AReaL-Hex]] (2511.00796) — MILP scheduler for hetero async RL, dense models, math workload. Methodological template; we extend.
- [[Papers/ReLibra|ReLibra]] (2605.08639) — routing-replay-guided load balancing across hierarchical network bandwidths. Closest hybrid (MoE + systems + hetero-bandwidth) but homog compute, within-replica.
- Missing Old Logits in Async Agentic RL (2605.12070) — semantic-mismatch repair for async, discusses R2/R3.
- Relax (2604.11554) — async RL engine supporting R3.

### Within-replica MoE RL — solved

- [[Papers/R3|R3]] (2510.11370) — inference-mask replay.
- [[Papers/Router-Aware IS|Router-Aware IS]] (2510.23027) — router-logit-rescaled IS.

### Hetero async RL on dense — solved

- [[Papers/AReaL-Hex|AReaL-Hex]] (2511.00796)
- [[Papers/RollArt|RollArt]] (2512.22560)
- StreamRL (cross-DC dense)
- ROLLMUX (phase-level multiplexing)

### Hetero MoE training — pretraining only

- [[Papers/HeterMoE|HeterMoE]] (2504.03871, EuroSys '25)
- [[Papers/Lazarus|Lazarus]] — elastic MoE pretraining
- LAER-MoE (ASPLOS '26)

### Production MoE RL stacks — no papers

- slime (THUDM / Z.ai) — Megatron + SGLang
- prime-rl (Prime Intellect) — FSDP2 + vLLM
- Miles (LMSYS) — slime-based on GB300
- Salesforce SFR-RL

### Stabilization formulation / survey

- Stabilizing RL with LLMs (2512.01374) — frames consensus on staleness + routing replay.

### Tool-execution / sandbox

- SWE-World (2602.03419) — Docker-free SWE agent environments.
- SWE-MiniSandbox (2602.11210) — container-free RL for SWE.

These two acknowledge sandbox overhead as a real bottleneck — validates the M4 angle.

## Out of scope / explicitly not claiming

- Final SWE-bench pass rate of trained model
- New RL algorithm (we use GRPO unchanged)
- New optimizer or new parallelism axis
- Dynamic / elastic cluster membership (Framing C from prior search, deferred)
- Repo-prefix KV cache (Framing A, deferred)
- VLA / multimodal RL workload (ruled out as primary)
- Megatron compatibility for B2 mechanisms (slime port is stretch / robustness experiment)

## Linked artifacts

- Task list — see auto-task list (B1: tasks 1–10, B2: tasks 11–17)
- Memory: [[paper-commitment]], [[research-related-work]], [[research-papers]]
- Paper notes: [[Papers/AReaL-Hex]], [[Papers/AReaL]], [[Papers/ReLibra]], [[Papers/R3]], [[Papers/Router-Aware IS]], [[Papers/prime-rl]], [[Papers/slime]]

## Decision log (key choices and why)

| Decision | Choice | Rationale |
|---|---|---|
| Framework | prime-rl | Ships Qwen3-30B-A3B-SWE recipe; FSDP2 easier to instrument than Megatron; positioned as research-modular |
| Training engine | FSDP2 | Matches prime-rl; easier instrumentation; perf gap vs Megatron acceptable at our scale |
| Inference engine | vLLM | Default for prime-rl; FP8 supported |
| RL algorithm | GRPO + R3 | prime-rl default; R3 prevents collapse; we don't innovate here |
| Model | Qwen3-30B-A3B | Production MoE; matches prime-rl recipe; LoRA acceptable for fine-tuning |
| Workload | mini-swe-agent + Docker + real tests | Bash-only agent surface; real test execution exposes structural properties Meta SWE-RL avoided |
| Evaluation framing | Throughput-not-quality | AReaL-Hex precedent; convergence wallclock infeasible at experimental scale |
| Headline contribution | MILP scheduler + 3 mechanisms | Single defensible system contribution; M4 as conditional scope expansion |
