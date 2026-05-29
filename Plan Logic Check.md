# Plan Logic Check

Companion to [[Paper Direction]]. Strips the plan to first principles to verify the logic holds end-to-end. Read this before committing engineering effort.

## 1. Workload (exact)

| Dim | Spec |
|---|---|
| Model | Qwen3-30B-A3B (30B total, ~3B active, 128 experts, MoE) |
| Fine-tuning | LoRA on top of pretrained base (full FT optional, OOM risk on workstation tier) |
| RL algorithm | GRPO with R3 (rollout routing replay) on |
| Agent | mini-swe-agent — bash-only tool surface (no separate file/test/run tools) |
| Sandbox | Docker container per episode, Prime Intellect Sandboxes or equivalent |
| Dataset | SWE-rebench-openhands trajectories (67K agent trajectories, 1,823 Python repos) |
| Reward | terminal binary, from running real project tests inside the sandbox |
| Async | rollout workers + train workers run concurrently with bounded staleness η |
| Multi-replica | K parallel rollout replicas, each running M concurrent episodes |
| Cluster | hetero, ≥2 compute tiers, ~8–24 GPUs total. Source treated as solved (academic shared or AWS) |

**Anchor numbers (to validate in Phase 1, not assumed for paper claims):**
- ~256 episodes per train step
- ~100 turns per episode
- ~1.5s GPU inference per turn
- ~30s typical test invocation, ~5 min full test suite
- 1–5 test invocations per episode
- Per-episode wallclock: minutes-to-tens-of-minutes

## 2. Existing tech to use (don't innovate at these layers)

| Layer | Component | Why we use it |
|---|---|---|
| Orchestration | prime-rl | Ships Qwen3-30B-A3B-SWE recipe; FSDP2 instrumentation easier than Megatron |
| Training engine | PyTorch FSDP2 | Native to prime-rl |
| Inference engine | vLLM (FP8) | Default for prime-rl |
| RL algorithm | GRPO | prime-rl default |
| MoE stability | R3 | Solved within-replica train-inference router KL |
| Agent | mini-swe-agent | Standard, scores 74%+ on SWE-bench verified |
| Sandbox | Prime Intellect Sandboxes (Docker exec) | Out-of-box |
| MoE parallelism | DTensor EP | Standard FSDP2 + DTensor composition |
| Weight broadcast | NCCL collectives | Standard |

**Nothing in this list is a research variable for us.** If we replace one, it's engineering, not novelty.

## 3. What's missing from existing research

The claim:

> **MoE × async × multi-replica × hetero compute × agentic workload — characterized for throughput, scheduled as a system — does not exist as a published paper.**

Double-check by neighbor:

| Neighbor | What it has | What it gives up |
|---|---|---|
| AReaL-Hex (2511.00796) | hetero × async × scheduler (MILP) × throughput | dense, math reasoning, no MoE, `C_I > C_T` assumption |
| ReLibra (2605.08639) | MoE × hetero × RL × system | bandwidth-hetero (not compute), single-replica routing replay |
| R3 (2510.11370), Router-Aware IS (2510.23027) | MoE × within-replica × algo stability | no system, no hetero, no multi-replica |
| Missing Old Logits (2605.12070) | async × agentic × semantic repair | no hetero, no MoE-specific scheduling |
| Relax (2604.11554) | async × R3 × omni-modal engine | no hetero scheduling, MoE not the headline |
| slime, prime-rl, Miles | MoE × async × multi-replica × production stack | no published throughput characterization, no hetero |
| HeterMoE (EuroSys '25), Lazarus, LAER-MoE | MoE × hetero × pretraining | not RL, no async, no rollout, no multi-replica drift |
| StreamRL, RollArt, ROLLMUX | hetero × async × RL | dense, math |
| Stabilizing RL with LLMs (2512.01374) | MoE + dense × stability formulation | survey, not a system |
| SWE-World, SWE-MiniSandbox (2602.03419, 2602.11210) | agentic × sandbox optimization | not MoE, not hetero, not scheduling |

**No cell covers all five dimensions at once.** Each adjacent paper gives up at least one.

**Honest caveats to verify:**
- Industry (Anthropic, OpenAI, Meta) almost certainly has internal work in this cell. Not published. Counts as zero for prior art but means our absolute claim is "first *published*."
- ReLibra full-text read is gating. If "hierarchical bandwidth" turns out to include compute-tier hetero, the gap narrows.
- Concurrent submissions to MLSys '27 / EuroSys '27 are blind to us. Move with reasonable pace.

## 4. Stage 1 — Profile what exactly

Six concrete questions, six charts. If we can't answer each from the data, the harness is incomplete.

| # | Question | Measurement | Chart |
|---|---|---|---|
| Q1 | Where does per-episode wallclock actually go? | Per-call wallclock + category tag (file_op / git / test / build / other / inference) | Stacked bar by category |
| Q2 | Is the GPU starved on this workload? | GPU busy % per rollout worker; sandbox queue depth over time | Time series, both axes |
| Q3 | Does AReaL-Hex's `C_I > C_T` regime hold for MoE? | Measure C_inference and C_train per step, both phases isolated | C_I/C_T ratio bar by tier |
| Q4 | How big is the weight broadcast cost, and how does it split? | Broadcast time decomposed: router bytes vs expert bytes, time per param group | Stacked bar per update |
| Q5 | How heavy-tailed are episodes? | Per-episode wallclock distribution | Histogram + CDF, with p50/p95/p99 marked |
| Q6 | Does cross-replica router KL drift exist and grow with tier gap? | Per-replica router-version logged; KL between same-prompt routing distributions across replicas | Time series, one line per replica pair |

**The six charts ARE the B1 paper.** If three of them show "no, this is the same as dense math RL," the contribution shrinks. If five+ show structural difference, B1 is publishable on its own.

**Sweep dimensions for the charts** (so each chart is a surface, not a point):
- Tier mix ratio
- Cluster size
- Rollout concurrency per worker
- EP degree
- Broadcast cadence

## 5. Stage 2 — Define the scheduling problem

### Inputs
- `D` — set of GPUs with per-GPU type (compute, HBM, interconnect)
- Workload spec — model, expected per-step token count, expected per-episode wallclock distribution
- `η` — data-staleness bound
- `ε` — router-KL bound (new vs AReaL-Hex)
- Budget — either GPU count fixed or $/hr fixed

### Decisions
| Variable | Meaning | New vs AReaL-Hex? |
|---|---|---|
| `D_T, D_I` | disjoint train/rollout GPU partition | Same |
| `σ, τ` | parallelism plans (TP, PP, DP, EP degrees per phase) | Same shape, but now `τ` includes EP for MoE |
| `e_{i,t}` | per-expert tier placement | **New** |
| `β_router, β_expert` | broadcast cadence per param group | **New** |
| `K` | rollout replica count | New (implicit in AReaL-Hex) |

### Constraints
- Memory: `MEM-CUMSUM(d) ≤ M_d` for all `d` (same as AReaL-Hex)
- Staleness: `staleness(replica_i) ≤ η` (same)
- **Router KL**: `KL(R_replica_i || R_trainer) ≤ ε` (new)
- **Tier-bandwidth-respecting EP**: all-to-all cost in `C_I` uses real per-pair bandwidth, not min-over-all (new)

### Objective
`min max(C_T, C_I)` — same shape as AReaL-Hex. `C_I` now includes EP all-to-all with tier-bandwidth costs and router broadcast amortization.

### Explicitly out of scope
- **Dynamic / elastic membership.** Static placement for this paper. Elasticity is a follow-up (was Framing C in prior search, deferred).
- **Sandbox CPU as resource** — see §6, conditional.

## 6. What happens when we add tool_execution

### What changes in the inputs
- New: `S` — set of sandbox-capable hosts with CPU cores and RAM per host
- New: expected tool-call latency distribution from Phase 1 (Q1)
- New: per-repo sandbox warmup cost

### What changes in the decisions
- New: `k_s` — sandbox concurrency per rollout worker
- New: `R_s` — per-repo sandbox pre-warm replication count
- New: sandbox→rollout-worker affinity (which CPU hosts serve which rollout workers)

### What changes in the constraints
- **Sandbox CPU capacity**: `sum(active sandboxes) ≤ |S| × cores/host`
- **Sandbox pre-warm budget**: `R_s × num_repos ≤ pre-warm memory budget`

### What changes in the objective
- `C_rollout` shifts from `C_inference / k_s` to `max(C_inference, C_tool) / k_s`

That `max(...)` is the entire structural change. When tools dominate, the scheduler optimizes sandbox concurrency. When GPU dominates, the scheduler optimizes inference parallelism. Same MILP, one new term, one new resource type.

### What case warrants adding it

Trigger from Phase 1 (Q2):

| Observation | Action |
|---|---|
| GPU busy % > 80%, queue depth ~0 | **Do not activate M4.** Tools are not the bottleneck. Note in writeup. |
| GPU busy % 60–80%, queue depth small but nonzero | **Borderline.** Run sweep with sandbox count varied; if changing sandbox count meaningfully shifts throughput, activate M4. |
| GPU busy % < 60%, queue depth > 0 steady-state | **Activate M4.** This is Regime B; GPU-only scheduler will make wrong decisions. |
| GPU busy % low because of OTHER reason (small batch, short episodes) | **Do not activate M4.** Diagnose and fix the actual cause. M4 is for sandbox starvation specifically. |

### Cost of being wrong

| Wrong direction | Consequence |
|---|---|
| Add M4 when not needed | Extra MILP complexity, no measured benefit, paper reads as overfit to non-issue |
| Skip M4 when needed | Scheduler over-provisions rollout GPUs (idle waiting for tools), hetero placement decisions are wrong on the rollout side, B2 throughput gains undercut by 20-40% silently |

The asymmetry favors **including M4 as a dormant variable in the MILP formulation from the start**, activating it as a measured mechanism only on the trigger. Inclusion costs a paragraph in the design doc; exclusion costs a redesign.

## 7. Logic check — does the chain hold end-to-end?

Three claims the paper relies on. Each must survive Phase 1:

| Claim | Survives if | Risk |
|---|---|---|
| **C1.** MoE × async × multi-replica × hetero × agentic is structurally different from prior cells | Q3, Q4, Q6 show non-trivial differences from dense math RL | If all three look the same as dense, paper becomes incremental |
| **C2.** A scheduler can exploit those differences | Q3 inverts `C_I/C_T` ratio; Q4 shows broadcast is a measurable chunk; Q6 shows cross-replica drift | If `C_I/C_T` regime matches AReaL-Hex's assumption and drift is negligible, B2 mechanisms have no headroom |
| **C3.** Tool execution either doesn't matter or can be added as an extension | Q1 + Q2 give a clean decision (Regime A → omit M4, Regime B → activate) | If Q1 shows a third regime (e.g., builds dominate uniformly, no scheduling can help), need a different mechanism |

If all three survive, the plan ships. If any breaks, the plan adjusts; see [[Paper Direction]] §Risks.

## 8. One-line summary

We use prime-rl + Qwen3-30B-A3B + mini-swe-agent + GRPO + R3 as the workload. We profile six things in Stage 1 to validate three structural claims. If validated, Stage 2 defines a hetero-aware MILP scheduler over expert placement, broadcast cadence, and (conditionally) sandbox concurrency. Same MILP shape as AReaL-Hex, new decision variables, new cost terms. Static placement only; dynamic/elastic is a follow-up.
