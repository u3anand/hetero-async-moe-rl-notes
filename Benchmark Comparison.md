# Benchmark Comparison — B1 (AutoPD-RL) vs B2 (TierShift)

The first runnable milestone for each candidate direction, side by side. Both run on **watgpu308**, both are **characterize-first** (the charts are the contribution and motivate the controller), both reuse the JSONL-telemetry philosophy. The decisive difference is **how much machinery must work before the first chart**.

- **Benchmark 1** = [[Initial Plan]] (full spec) — for [[Research Plan]] (AutoPD-RL).
- **Benchmark 2** = §B2 of [[Research Plan 2]] (TierShift) — specified below.

## Side by side

| | **Benchmark 1 — PD-demand profiling** | **Benchmark 2 — hotness×tier profiling** |
|---|---|---|
| **Thesis it motivates** | static PD split mis-fits non-stationary rollout demand | static expert→tier placement mis-fits non-stationary expert hotness |
| **Workload** | full **async agentic-RL rollout loop** (train + rollout + env) | **inference serving only** — replay a request trace |
| **Moving parts** | prime-rl (trainer FSDP2+LoRA GRPO · SampleBuffer · bounded-staleness) + mini-swe-agent + **udocker sandbox rewards** + vLLM **PD-disaggregated** + weight sync | vLLM/SGLang **EP serving** + per-expert telemetry + trace replayer |
| **What "make it work" costs** | high — trainer⇄rollout⇄sandbox all must run, survive preemption, reach steady state | low — serve a fixed checkpoint + replay a trace; no training, no agent, no reward, no weight sync |
| **GPUs (bring-up)** | 6 (4 rollout + 2 train) | **2** (1 fast Ada + 1 slow A6000) |
| **Model** | OLMoE-1B-7B → Qwen3-30B-A3B | OLMoE-1B-7B (64 exp) → Qwen3-30B-A3B (128 exp) |
| **Core signal** | per-turn prefill:decode demand over wall-clock | per-expert token load over wall-clock, per tier |
| **Headline chart** | **P:D-demand-over-time vs static split** (non-stationarity) | **expert-hotness-over-time vs static tier placement** (non-stationarity) |
| **Kill-question** | is PD demand non-stationary enough to beat any static split? | is hotness non-stationary **and** does it churn *slower* than PCIe migration? (Q3) |
| **Fail-soft** | if PD demand is stationary → no controller, B1 still a characterization paper | if churn > migration speed → "replication-only" finding (still novel vs homogeneous CRAFT) |
| **Hard dependencies** | udocker reward sandboxes (CPU-bound, 32-core contention), preemption/requeue, checkpoint resume | per-expert load exposure in the serving fork; a migrate/hot-swap hook |
| **Rough effort** | the larger lift — most of the work is the loop standing up | the smaller lift — ~a serving harness + telemetry |

## Benchmark 2 — concrete spec

**Goal:** serve a fixed MoE across both tiers under expert parallelism, replay a *mixed, time-varying* request trace, and emit Q1–Q7 (see [[Research Plan 2]] §2) — the hotness-×-tier profile that motivates (or kills) the TierShift controller.

**Steps (each gated GO/NO-GO):**
0. **Recon** — `nvidia-smi -L` tier map (fast Ada vs slow A6000); confirm PCIe topology (no NVLink → migration goes over PCIe). *Gate:* tier map + PCIe path confirmed.
1. **Serve spike** — vLLM/SGLang **EP**: pin some OLMoE experts to 1 Ada, the rest to 1 A6000; serve a short trace. *Gate:* cross-tier EP serves; per-expert token counts + per-GPU busy% extractable from the fork.
2. **Trace replayer** — assemble a mixed stream (chat → code → math segments, from ShareGPT/LMSYS + a code/math slice) that *induces* hot-set drift; replay at controlled rate. *Gate:* measurable shift in the per-layer hot set across segments (Q1/Q6 anchor).
3. **Migration micro-bench** — measure expert-weight size + tier↔tier PCIe transfer time; measure hot-set turnover rate. *Gate (the kill-test):* migration time vs churn timescale recorded → decides migrate-vs-replicate-only (Q3/Q4).
4. **Static-placement straggler measurement** — fix an Aurora-style placement, replay the drifting trace, log per-tier busy% + slow-tier straggler wait as the hot set moves off/onto each tier. *Gate:* static placement shows measurable straggler/idle under drift (Q2).
5. **Instrumentation + charts** — Q1–Q7 → JSONL → charts; headline = hotness-over-time vs static placement (Q1/Q2) + migration-cost vs hotness-timescale (Q3).

**Telemetry (two JSONL streams):**
- `expert.jsonl` — per (layer, expert, time-window): `token_load`, `tier`, `is_replica`; routing snapshot for the predictor.
- `system.jsonl` — ~1 Hz: per-GPU `sm_active%`, `dram_active%`, cross-tier all-to-all bytes, per-tier straggler wait, queue depths.

**Definition of done (B2):** on watgpu308, a fixed MoE serves across fast+slow tiers under EP with a drifting mixed trace, emits Q1–Q7 to charts — headlined by **expert-hotness-over-time vs static placement** and the **migration-cost vs hotness-timescale** kill-chart. Anchor numbers recorded (imbalance ratio swing, slow-tier straggler %, expert-migration ms over PCIe, hot-set turnover rate, predictor lead time). This proves the harness + motivates (or bounds) TierShift; the Qwen3-30B-A3B phase reuses it.

## The decision this comparison informs
B2 is the **cheaper, faster-to-first-result** milestone (no trainer/agent/sandbox, 2 GPUs not 6) — Benson's feasibility argument made concrete. B1 is the **larger lift** but is already partway built (the `code/b1/` harness, forks, and run logs exist). The trade: **sunk progress on B1** vs. **lower remaining cost + cleaner "serving, not RL" framing on B2.**

## Links
[[Research Plan]] · [[Research Plan 2]] · [[Initial Plan]] · [[Aurora]] · [[MegaScale-Infer]]
