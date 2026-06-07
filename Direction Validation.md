# Direction Validation — AutoPD-RL

Literature validation of the pivot from the MoE-hetero-MILP thesis to **automatic prefill/decode (PD) allocation for agentic RL rollouts on heterogeneous GPUs**. Verified against primary sources **2026-06-03/04** (parallel research agents fetching arXiv/GitHub, not search snippets). Companion to [[Research Plan]] · paper notes in `papers/`.

> ⚠️ **Epistemic caveat.** Several load-bearing papers carry **post-training-cutoff arXiv IDs** (2602–2605 = Feb–May 2026): [[ROSE]], [[Heddle]], [[HexAGenT]], [[TokenScale]], [[DOPD]], [[Freshness-Aware-PER]]. PDFs were downloaded and are real in this environment, but precise quotes/authors for post-cutoff papers carry fabrication risk. **Re-read the actual PDFs before building positioning on them** — especially [[RollArt]] (foundation) and [[HexAGenT]] (closest threat).

## The wedge (one sentence)

> RollArt demonstrates disaggregated, hardware-affinity PD for agentic RL but configures the prefill/decode instance split **manually/statically** and explicitly leaves **automatic configuration as future work**. Serving systems already do dynamic PD autoscaling, but assume **static weights + SLO objective** — assumptions RL rollout breaks. AutoPD-RL = **automatic/dynamic PD-pool allocation under RL-specific dynamics** (weights updated every step, batch-synchronous long-tail, train/rollout contention, objective = *fresh complete trajectories per training deadline*).

## What verified TRUE

| Paper | ID | Verdict | Role |
|---|---|---|---|
| [[RollArt]] | 2512.22560 | all 6 claims TRUE | **Baseline-to-beat**; names our gap; SampleBuffer idle = 62% of iter |
| [[ROSE]] | 2605.06534 | all 3 TRUE | Spare-GPU elastic rollout (different lever) |
| [[Heddle]] | 2603.28101 | 1–7 TRUE; **#8 FALSE** | Trajectory sched; PD-sizing **explicitly orthogonal** |
| [[HexAGenT]] | 2605.16637 | verified | Closest PD-routing neighbor; **fixed pools + serving SLO** |
| [[RollPacker]] | 2509.21009 | TRUE | Tail batching (batch-round granularity) |
| [[EARL]] | 2510.05943 | TRUE | Dynamic parallelism under long context |
| [[ROLL]] | 2506.06122 | TRUE | Base lib; `AutoDeviceMapping` = manual knob |
| [[Freshness-Aware-PER]] | 2604.16918 | TRUE | Freshness objective (algorithmic) |

## Two corrections to the prior (handoff) plan

1. **Heddle is NOT a threat to AutoPD-RL.** Its adaptive resource manager tunes **per-trajectory model-parallelism within a fixed pool**; the paper calls PD-disaggregation "orthogonal." The earlier pivot's strongest stated reason was partly a misread. Heddle threatens *generic trajectory scheduling* (already retired), not PD sizing.
2. **The real threats were unlisted:**
   - **Serving-side PD autoscaling is crowded and mature** — [[TokenScale]] (token-velocity + convertible decoders), [[DOPD]], [[Arrow]], [[DynaServe]] all do dynamic PD-ratio control. So "dynamically size PD pools" is **not novel per se**. Novelty must live in the **RL dynamics + objective**.
   - **[[HexAGenT]]** (not in the handoff) is the closest scheduling neighbor — but routes over fixed pools for serving SLOs, leaving sizing + the RL objective open.

## Why serving solutions don't port (the defensible novelty)

Serving PD-autoscalers assume **static weights + SLO-driven request streams**. RL rollout differs on three axes:
1. **Weights updated every step** → KV/instance reuse + warm-start assumptions break; cross-cluster weight sync ([[RollArt]] Mooncake) interacts with PD layout.
2. **Batch-synchronous long-tail** → >90% of runtime in tail; objective is *batch completion*, not per-request latency.
3. **Train/rollout contention + freshness** → downstream goal is fresh complete trajectories before the trainer's `get_batch` deadline ([[RollArt]] SampleBuffer 62% idle; cf. [[Freshness-Aware-PER]]), not request latency.

## Strategy

- **Cite [[RollArt]] as the static baseline to beat**; quote its future-work admission.
- **Borrow mechanisms** from [[TokenScale]]/[[DynaServe]] (token-velocity, convertible decoders, per-batch P:D), then **show they fail unmodified** under RL dynamics.
- **Carve out [[HexAGenT]]** (fixed pools, serving SLO) and **[[Heddle]]** (MP-tuning, orthogonal).
- The retired **SampleBuffer-freshness** idea returns as the **objective function**, not as a trajectory-scheduling mechanism (avoids Heddle overlap).
- **Move fast** — the gap is acknowledged by a leading group (Alibaba/ROLL) and likely to close within a release cycle.

## Open risks to resolve next

- Confirm [[HexAGenT]] / [[RollArt]] from PDFs (post-cutoff) before locking the related-work delta.
- Is the **SWE-predictor** sub-angle (repo/test/build/cache signals → PD-pressure class) a real measurable signal or assumed? → a B1 profiling result, not an assumption.
- Dropped-but-on-thesis existing notes worth possibly restoring from git: **AReaL-Hex** (async RL over heterogeneous GPUs), Mooncake (cross-cluster weight transfer), vLLM/SGLang (serving substrate), TaiChi (PD aggregation unifier).
