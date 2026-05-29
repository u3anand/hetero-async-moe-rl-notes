# Paper Direction — v2

Final consolidated plan. Supersedes [[Plan Logic Check]] and earlier revisions of [[Paper Direction]].

Date: 2026-05-29. Working title: *Throughput-Optimal Asynchronous MoE RL on Heterogeneous GPU Clusters for Agentic Workloads*.

---

## 1. Workload

Exact spec. Nothing here is a research variable.

| Dim | Spec |
|---|---|
| Model | Qwen3-30B-A3B (30B total params, ~3B active per token, 128 experts) |
| Fine-tuning mode | LoRA on pretrained base (full FT optional) |
| RL algorithm | GRPO with R3 (rollout routing replay) enabled |
| Agent harness | mini-swe-agent — bash is the only tool surface |
| Sandbox | Docker container per episode (Prime Intellect Sandboxes or equivalent) |
| Dataset | `nebius/SWE-rebench-openhands-trajectories` (67K trajectories, 1,823 Python repos) |
| Reward | Terminal binary, from running the project's actual test suite |
| Orchestration | prime-rl |
| Training engine | PyTorch FSDP2 |
| Inference engine | vLLM (FP8 supported) |
| MoE parallelism | DTensor EP via FSDP2 composition |
| Concurrency model | Asynchronous multi-replica rollout; bounded staleness `η` |
| Default batch | 256 episodes / train step (32 prompts × G=8 rollouts) |

**Anchor numbers to validate in Stage 1 (not assumed for paper claims):**
- ~100 turns per episode
- ~1.5s GPU inference per turn (Qwen3-30B-A3B on H100)
- Bash calls: ~95% fast file ops (<100ms), ~5% test/build runs (30s–5min)
- ~1–5 mid-episode test runs per episode (agent-initiated)
- 1 end-of-episode reward eval per episode (system-initiated, full test suite, 30s–5min)
- Per-episode wallclock: 5–30 min, heavy-tailed

---

## 2. Benchmark Stage (Stage 1)

**Goal:** produce the workload characterization that motivates the scheduler and provides the cost inputs the scheduler consumes. Six questions, six charts. Each chart is a surface across the sweep dimensions, not a single point.

### Six characterization questions

| # | Question | Measurement | What it determines |
|---|---|---|---|
| Q1 | Where does per-episode wallclock go? | Per-tool-call wallclock + category tag (file_op / git / mid_test / build / reward_eval / inference) | Is tool I/O bimodal? Where to invest mechanism effort |
| Q2 | Is the GPU starved on this workload? | GPU busy % per rollout worker; sandbox queue depth over time | Activates or shelves M4 (tool-execution-aware scheduling) |
| Q3 | Does AReaL-Hex's `C_I > C_T` regime hold on MoE? | Isolate `C_inference` and `C_train` per step | Confirms or refutes the central scheduling assumption |
| Q4 | How big is weight broadcast cost, and how does it split? | Broadcast time per param group (router bytes vs expert bytes) | Sizes M1 (decoupled broadcast) win |
| Q5 | How heavy-tailed are episodes? | Per-episode wallclock CDF with p50 / p95 / p99 | Drives concurrency formula and MILP cost terms |
| Q6 | Does cross-replica router KL drift grow with tier-speed gap? | Per-replica router-version logged; KL between rollout-version routing dists across replicas | Sizes M3 (cross-replica routing replay) win |
| Q7 | Train-phase MFU + rollout-phase HBM bandwidth utilization (supplementary) | Train: achieved FLOPs / theoretical peak per train GPU. Rollout: HBM bandwidth utilization (MFU is structurally low for inference; HBM is the meaningful number) | Sanity check that we are not leaving kernel-level perf on the table. Not a headline metric — AReaL-Hex precedent. |

### Sweep dimensions (each chart is a surface, not a point)

- Tier mix ratio (100/0, 75/25, 50/50, 25/75, 0/100)
- Cluster size (8, 16 GPU)
- Rollout concurrency per worker (`N`)
- EP degree
- Broadcast cadence

### Distribution-aware framing

Charts report **p50 / p95 / p99**, not just means. Heavy-tailed episode wallclock and bimodal tool-call latency mean mean-based reporting hides what matters. The scheduler in §4 consumes these distributions (or tail quantiles) directly as cost inputs, not just means.

### What success looks like

If five of six charts show non-trivial divergence from prior dense / math RL characterizations, B1 is publishable standalone (MLSys workshop / characterization track). If included as §3 of B2, it's the load-bearing motivation. Either way: **the charts ARE the contribution at this stage.**

### Throughput-not-quality stance

We measure throughput in steady-state windows (~4 hours per config). We do NOT measure final RL quality. Precedent: AReaL-Hex headlines "1.31–1.50× throughput, 1.46× cost reduction" with no convergence quality claim. We adopt this stance to keep scope feasible.

---

## 3. RL Problem (the loop, for grounding)

Async multi-replica RL with GRPO. Two roles running concurrently:

### Rollout side (most of the cluster)

Continuously generates trajectories. For each in-flight episode:

```
while episode not done:
    state = current conversation
    action = vLLM.generate(state, weights=θ_rollout)   # GPU, ~1.5s
    obs = sandbox.exec(action)                          # CPU, ~50ms–30s
    append (state, action, obs) to trajectory

# Episode end
reward = sandbox.run_test_suite()                       # CPU, 30s–5min
trajectory.reward = reward
push trajectory to shared buffer
```

K replicas run independently, each with M concurrent episodes. They never stop except for weight broadcast adoption.

### Train side

```
loop:
    wait until buffer has ≥ batch_size trajectories
    batch = buffer.drain(batch_size)
    # GRPO advantage
    for each prompt's G rollouts:
        advantages = (rewards - mean(rewards)) / std(rewards)
    # Gradient step
    loss = -E[advantage × log π(action|state)] + β · KL(π || π_ref)
    θ_new = optimizer.step(grad(loss))
    broadcast θ_new to rollout workers
```

### Asynchrony invariants

- Rollout episodes run at whatever `θ_t` the worker had when the episode *started*
- An episode finished at `θ_t` may be consumed for training at `θ_{t+k}`; `k` is the staleness
- `η` bounds staleness. Backpressure pauses new episodes if buffer would create stale rollouts beyond `η`
- Weight broadcast is the only point rollout pauses briefly (seconds-to-minutes for the swap)

### Why rollout dominates

| Phase | Wallclock per training step (small cluster) | FLOPs share |
|---|---|---|
| Rollout (inference + tools) | 15–30 min | ~80–90% |
| Training (forward + backward + opt) | 1–5 min | ~10–20% |
| Weight broadcast | 30s–2 min | ~0% (bandwidth-bound) |

Implication: GPU-side optimizations on rollout > training. MoE flips part of this on hetero — see Q3 above.

---

## 4. Scheduling Problem and Solution

### Static placement + dynamic runtime control

Two-tier architecture for the scheduler:

| Tier | Cadence | Decisions | Mechanism |
|---|---|---|---|
| **Coarse — static** | Once per run | GPU partition, expert placement, parallelism degrees, replica count | MILP solve |
| **Fine — dynamic** | Every few seconds | Broadcast cadence, rollout concurrency, episode timeout, backpressure thresholds | Feedback rules |

The MILP outputs setpoints; lightweight controllers maintain them. **No MILP re-solve during the run.** Full elastic scheduling is deferred to future work.

### Static MILP (extends AReaL-Hex)

**Inputs:**
- `D` — heterogeneous GPU set with per-GPU type (compute, HBM, interconnect)
- `S` (dormant) — sandbox CPU host pool
- Workload spec: model, expected per-step token count, **per-episode wallclock distribution from B1**, **per-tool-call latency distribution from B1**
- `η` — data staleness bound
- `ε` — router KL bound (new vs AReaL-Hex)

**Decision variables:**

| Variable | Meaning | Source |
|---|---|---|
| `D_T, D_I` | Train / rollout GPU partition (disjoint) | AReaL-Hex |
| `σ, τ` | Parallelism plans (TP/PP/DP/EP per phase) | AReaL-Hex |
| `K` | Rollout replica count | AReaL-Hex implicit, explicit here |
| `e_{i,t} ∈ {0,1}` | Per-expert tier placement | **New** |
| `β_router, β_expert` | Broadcast cadence per param group | **New** |
| `k_s, R_s` (dormant) | Sandbox concurrency, per-repo prewarm replication | **New, conditional on B1 Q2** |

**Constraints:**
- Per-GPU memory (same as AReaL-Hex)
- Staleness: `staleness(replica_i) ≤ η`
- **Router KL: `KL(R_replica_i || R_trainer) ≤ ε`** (new)
- **Tier-bandwidth-respecting EP all-to-all cost** (new — no min-over-all assumption)
- Sandbox CPU capacity (dormant — activates if M4 triggers)

**Objective:**
- `min max(C_T, C_I)` — same shape as AReaL-Hex
- `C_I = max(C_inference, C_tool) / k_s` if M4 active; else `C_inference / k_s`
- **Cost terms use distributional inputs (p95) from B1, not means**, to avoid under-provisioning for heavy-tailed workloads

### Dynamic runtime controllers

Maintain MILP setpoints reactively. Each controller is a simple feedback rule:

| Knob | Trigger signal | Update rule |
|---|---|---|
| Router broadcast | Cross-replica router KL approaches `ε` | Broadcast router immediately when KL > 0.8 · ε |
| Expert broadcast | Accumulated expert grad norm threshold | Broadcast experts at threshold or fallback cadence |
| Rollout concurrency `N` | GPU busy %, sandbox queue depth | Raise N if GPU busy < 80% and queue ~0; lower if queue growing |
| Episode timeout | Sandbox queue depth, GRPO group-ready lag | Force kill at p95 if queue growing or group lag > 2× expected |
| Backpressure | Staleness headroom | Tighten if observed staleness approaches `η` faster than predicted |

### Three composable mechanisms the scheduler controls

| # | Mechanism | What it does | Static or dynamic | Expected impact |
|---|---|---|---|---|
| M1 | Decoupled router/expert broadcast | Router (~5% of params) high cadence; experts (~95%) low cadence | Both: MILP picks cadence setpoints; controller adapts to KL | Large; broadcast is measurable chunk of step time |
| M2 | Tier-aware EP all-to-all | Expert placement and all-to-all routing avoid lock-step to slow tier | Static MILP only | Likely the largest single win |
| M3 | Cross-replica routing replay with version tags | Extends R3 with per-token router-version metadata; train side reconciles cross-replica drift | Static (protocol) + dynamic (replay engine) | Supports useful-token throughput; stability |

### Optional fourth mechanism (conditional)

**M4 — Tool-execution-aware co-scheduling.** Adds sandbox CPU as a first-class resource type in the MILP. **Dormant variable in formulation from day one; activated only if B1 Q2 shows GPU starvation.** Activation flag flips after measurement, not redesign.

### Why this is novel

Five-dimensional cell — **MoE × async × multi-replica × hetero compute × agentic workload — characterized for throughput, scheduled as a system** — is not in any published paper. Each neighbor gives up at least one dimension. Verified against AReaL-Hex (dense), ReLibra (homog compute, single-replica), R3 / Router-Aware IS (algo only), Relax / Missing Old Logits (no hetero), HeterMoE / Lazarus / LAER-MoE (pretraining), production stacks (no published throughput characterization), and SWE-World / SWE-MiniSandbox (sandbox optimization but not MoE / hetero / scheduling).

### Ablation matrix for evaluation

For each of three baseline configs (homog-fastest, homog-slowest, naive-hetero):
- prime-rl baseline (vanilla)
- + M1 (decoupled broadcast)
- + M3 (cross-replica routing replay)
- + M2 (tier-aware EP)
- Full system (M1 + M2 + M3 + scheduler with feedback)
- Full system + M4 (only if B1 Q2 triggers)

**Headline claim:** full system on naive-hetero recovers ≥80% of homog-fastest throughput at hetero cost.

---

## 5. Resources — GPU cluster planning

### Assumption

Heterogeneous cluster, ≥2 compute tiers, 8–24 GPUs. Source is interchangeable: shared academic infra (Waterloo watgpu) if accessible at the needed concurrency, AWS otherwise. We default to **AWS for planning purposes** because it's the lower-risk option for sustained access and contestable allocations.

### Goal: cheap, sufficient

We want the cheapest mix that lets the experiments run. The benchmark and system are interesting at any scale; we don't need flagship hardware. Cheaper tiers also strengthen the hetero story because they make the bandwidth/compute gap more pronounced.

### AWS instance shortlist

On-demand pricing (us-east-1, approximate; spot is 50–70% cheaper):

| Instance | GPUs | Tier | On-demand $/hr | Spot $/hr (est) | Use |
|---|---|---|---|---|---|
| `p5.48xlarge` | 8× H100 80GB SXM | Top (Hopper) | ~$98 | ~$30 | Reference / homog-fastest baseline |
| `p4d.24xlarge` | 8× A100 40GB SXM | Middle (Ampere SXM) | ~$32 | ~$10 | Fast tier alternative |
| `p4de.24xlarge` | 8× A100 80GB SXM | Middle (Ampere SXM) | ~$40 | ~$13 | Fast tier, more memory |
| `g6e.48xlarge` | 8× L40S 48GB | Lower (Ada workstation) | ~$30 | ~$10 | **Cheap slow tier — PCIe, real hetero gap** |
| `g6.48xlarge` | 8× L4 24GB | Lowest | ~$13 | ~$4 | Too memory-constrained for 30B model |
| `g5.48xlarge` | 8× A10G 24GB | Low (Ampere workstation) | ~$16 | ~$5 | Tight on memory; possible with aggressive LoRA |

**Recommended hetero pair for the paper:**
- Fast tier: **8× A100 80GB (`p4de.24xlarge`)** — Hopper-class capability without Hopper price; supports full Qwen3-30B-A3B sharding cleanly
- Slow tier: **8× L40S (`g6e.48xlarge`)** — Ada workstation, PCIe-only, real bandwidth gap; same VRAM bucket but very different interconnect
- Hetero shape: 1× p4de + 1× g6e = 16 GPU, 2 tier, ~$70/hr on-demand, ~$23/hr spot

This pair maximizes the hetero "story" (Hopper-class SXM/NVLink vs Ada workstation/PCIe) at the lowest total cost.

### "Naive hetero" baseline shape — homog train, hetero rollout

prime-rl will likely not start cleanly across mixed-tier FSDP shard groups (DTensor expects uniform device class per submesh). The "naive hetero" baseline therefore uses **homogeneous training pool, heterogeneous rollout pool** — matching AReaL-Hex's disjoint `D_T`, `D_I` assumption:

- Train pool: 8× A100 (homog within train)
- Rollout pool: 4× A100 + 4× L40S (hetero within rollout, separate inference processes per tier)

This avoids the hardest plumbing issue (FSDP2-across-hetero training GPUs) while still exposing the rollout-side hetero degradation that B2 must fix. The mixed-tier *training* MILP variables stay in the formulation for completeness but are exercised only in scope-stretch experiments.

### Alternative: smaller, cheaper for B1 only

If B1 needs scoping before committing to the full pair, a single `g6e.48xlarge` (~$30/hr on-demand) gets you 8× L40S — enough to bring up prime-rl + Qwen3-30B-A3B-SWE end-to-end and characterize a homog-slow baseline. ~$120 buys a 4-hour scoping run. This is the cheapest path to validating the workload runs at all.

### Compute budget estimate

| Phase | Compute hours | Approx $ (on-demand) | Approx $ (spot) |
|---|---|---|---|
| Stage 1 scoping (single config) | 4 hr × 1 config | ~$120 | ~$40 |
| Stage 1 sweep (5 configs × 3 reps × 4 hr) | 60 hr | ~$4,200 | ~$1,400 |
| Stage 2 design (no compute) | 0 | $0 | $0 |
| Stage 3 implementation (light testing) | ~20 hr | ~$1,400 | ~$500 |
| Stage 4 evaluation (3 configs × 6 ablations × 3 reps × 4 hr) | 216 hr | ~$15,000 | ~$5,000 |

**Total realistic budget: $20K on-demand, $7K on spot.** Tractable on most academic / research grants.

### Cost reduction levers (in order of effort)

1. **Use spot instances** — 50–70% discount. Risk: preemption during a run, but RL runs naturally checkpoint; loss is recoverable.
2. **Use Lambda Labs, RunPod, Crusoe, CoreWeave** — often 30–50% cheaper than AWS for the same hardware. RunPod has community-priced A100s under $1/hr.
3. **LoRA-only fine-tuning** — already in spec; reduces train-side memory pressure, may allow tighter cluster sizes.
4. **Smaller MoE for B1 only** — DeepSeek-V2-Lite (16B / 2.4B active) or OLMoE-1B-7B halves the memory footprint. Workload characterization story still transfers; only the absolute numbers shift.
5. **Drop replication count from 3 to 2 reps** — saves a third of eval compute; tradeoff is noisier results on heavy-tailed workload.

### Watgpu fallback / supplement

If watgpu access is granted for sustained reservations (4-hr+ holds on 8–16 GPUs across two tiers), use it for B1 scoping and bring-up. Move to AWS for the eval sweep where statistical power matters. Hybrid use is fine; the methodology doesn't depend on the source.

### What we do NOT need

- We do not need H200 NVL. The hetero story is sharper with cheaper Ada workstation cards anyway.
- We do not need GB300 or B200. Out of budget and orthogonal to the contribution.
- We do not need 1000+ GPU scale. AReaL-Hex evaluated on 48 GPUs; we operate in the same regime.
- We do not need to demonstrate convergence (see throughput-not-quality stance above), so no multi-week training runs.

---

## 6. What's explicitly out of scope

- New RL algorithm (we use GRPO unchanged)
- New optimizer
- New parallelism axis (use standard TP/PP/DP/EP)
- Megatron-side implementation (FSDP2 only; slime port held as stretch)
- Convergence quality claims (throughput-only headline)
- Dynamic MILP re-solve (static placement only)
- Elastic / dynamic cluster membership (Framing C — deferred)
- Repo-prefix KV cache (Framing A — deferred)
- VLA / multimodal RL (ruled out)
- p99 latency below the heavy-tailed regime (we report p95; p99 needs more data than budget allows)

---

## 7. Linked artifacts

- Task list (auto-managed): B1 = #1–10, B2 = #11–17
- Memory: [[paper-commitment]], [[research-related-work]], [[research-papers]]
- Paper notes (vault): [[Papers/AReaL-Hex]], [[Papers/ReLibra]], [[Papers/R3]], [[Papers/Router-Aware IS]], [[Papers/prime-rl]], [[Papers/slime]], [[Papers/HeterMoE]], [[Papers/Lazarus]]
- Superseded: [[Plan Logic Check]], earlier revisions of [[Paper Direction]]
