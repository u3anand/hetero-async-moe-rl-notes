# Papers Index

Curated literature for **AutoPD-RL** — automatic prefill/decode allocation for agentic RL rollouts on heterogeneous GPUs. See [[Direction Validation]] for the verified landscape and [[Research Plan]] for the thesis.

> Curated 2026-06-04. The previous MoE-internals-heavy set (DeepSpeed-MoE, FastMoE, Tutel, HeterMoE, …) was removed when the project pivoted away from the MoE-hetero-MILP thesis — recoverable from git history if needed.

## Agentic RL Systems — the wedge's direct neighbors

- [[RollArt]] — 2025 — arxiv:2512.22560 — **baseline-to-beat; names our gap (manual PD config = future work)**
- [[HexAGenT]] — 2026 — arxiv:2605.16637 — **novelty boundary**: PD routing over *fixed* pools, serving SLOs
- [[Heddle]] — 2026 — arxiv:2603.28101 — **novelty boundary**: trajectory scheduling; PD-sizing explicitly *orthogonal* (NOT a threat)
- [[ROSE]] — 2026 — arxiv:2605.06534 — elastic rollout on spare serving GPUs (different lever)
- [[RollPacker]] — 2025 — arxiv:2509.21009 — long-tail via tail batching (batch-round granularity)
- [[EARL]] — 2025 — arxiv:2510.05943 — dynamic parallelism under long/dynamic context
- [[ROLL]] — 2025 — arxiv:2506.06122 — base library; `AutoDeviceMapping` = the manual knob we automate
- [[Freshness-Aware-PER]] — 2026 — arxiv:2604.16918 — freshness objective (algorithmic, not resource)
- [[RLix]] — 2025 — github — multi-job GPU sharing (adjacent axis)

## Serving-side Prefill/Decode Disaggregation — prior art to distinguish from

**Static / founding:**
- [[DistServe]] — 2024 — arxiv:2401.09670 — offline PD placement search, SLO objective
- [[Splitwise]] — 2023 — arxiv:2311.18677 — offline pool sizing + limited dynamic mixed pool

**Dynamic PD autoscaling (the "mechanism is already solved for serving" cluster):**
- [[TokenScale]] — 2025 — arxiv:2512.03416 — token-velocity autoscaling + convertible decoders
- [[DOPD]] — 2025 — arxiv:2511.20982 — real-time-load-driven dynamic P/D ratio
- [[Arrow]] — 2025 — arxiv:2505.11916 — stateless instances + elastic PD pools
- [[DynaServe]] — 2025 — arxiv:2504.09285 — per-batch P:D ratio / batch / context tuning

## MoE Serving on Heterogeneous GPUs — candidate-pivot literature (exploratory, NOT AutoPD-RL)

> Added 2026-06-07 while scoping a possible pivot toward **online expert
> placement/replication/migration for MoE serving on heterogeneous GPU tiers** (Benson's
> inference-over-RL steer). These are *boundaries to carve against*, not AutoPD-RL neighbors. The
> static cell is taken; the surviving wedge is the **online/non-stationary** axis both papers punt on.

**The occupied cells (boundaries to carve against):**
- [[GEM]] — 2026 — arxiv:2605.19945 — **THE closest / most dangerous**: GPU-variability-aware expert→GPU *mapping*, same straggler motivation, consistent-vs-temporal experts. But **static** (names "no online adaptation" as its limit), **intra-generation** variability (7.7–27.7%, not a 2–4× tier gap), **one-expert-one-GPU** (no replicate/split), latency-only. Qwen3-30B-A3B → only 1.5% (uniform routing). The GEM-style static map is a **required B2 baseline**.
- [[Aurora]] — 2024 — arxiv:2410.17043 — **static-core owner**: optimal hetero expert placement + comm scheduling (colocated-on-hetero = NP-hard). Offline only; names "online dynamic rebalancing" as the gap.
- [[MegaScale-Infer]] — 2025 — arxiv:2504.02263 — **production heavyweight (ByteDance, ~10k GPU)**: attention/expert disaggregation across H20+L40S tiers. Static placement + greedy historical hot-expert replication; no online migration.
- [[HarMoEny]] — 2025 — arxiv:2506.12417 — **closest *dynamic* neighbor**: online token redistribution + async expert prefetch vs load imbalance. But **homogeneous** (rebalances to *idle*, not *faster*, GPUs — no tier-speed model); throughput/TTFT, not p99/$. A required B2 baseline.

**The perimeter (fences the wedge — none occupy MoE × hetero × serving × online):**
- [[HeterMoE]] — 2025 — arxiv:2504.03871 — same attention/expert split, but **training**; static. Serving absent from scope.
- [[Helix]] — 2024 — arxiv:2406.01566 — hetero serving via **max-flow + MILP**, but **dense** (LLaMA); placement static (run once).
- [[HexGen-2]] — 2025 — arxiv:2502.07903 — **PD-disaggregated** hetero serving, but **dense** (OPT/Llama-2); offline placement-style.

**The mechanism anchors (the reactive-cache idea TierShift builds on — Layer 0):**
- [[Toward-Efficient-MoE-Inference]] — NeurIPS'24 — **Expert Buffering**: reactive cache of hot experts, exploits temporal locality. *Reacts, doesn't predict* — resolves "online prediction is weird." But memory-caching GPU↔CPU (stall-on-miss), not compute-tier load-balancing. The naive 2-GPU port = TierShift's Layer-1 baseline.
- [[SwapMoE]] — 2024 — arxiv:2308.15030 — dynamic resident "Virtual Experts" + tunable memory, but **approximate** (accuracy drop) + single-device. The "approximate" corner TierShift declines.

**The opening (post step-(c) refinement):** Premise *confirmed* — expert hotness is non-stationary, domain/phase-dependent, measured on ShareGPT/LMSYS with Mixtral/Phi-3.5 (Imbalance Ratio swings 1.43–2.28). **But** that maturity means a dynamic crowd already exploits it on **homogeneous** HW (predictive prefetch / replication / load-forecasting): "Prediction Is All MoE Needs" (2404.16914), "Patterns behind Chaos" (2510.05497), PROBE (2602.00509), CRAFT (2603.28768), HarMoEny (2506.12417), MoE-Infinity (2401.14361). So the surviving cell narrows to: **online expert placement + *migration across heterogeneous tiers* under a tier-cost model** — homogeneous-dynamic crowd makes "where" trivial; Aurora decides "where" only offline. Novel *characterization* = the **tier × non-stationarity interaction**, not non-stationarity alone.
