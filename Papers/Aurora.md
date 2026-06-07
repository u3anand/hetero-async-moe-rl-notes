---
paper_id: "arxiv:2410.17043"
title: "Aurora: Optimizing MoE Inference Time Combining Model Deployment and Communication Scheduling"
year: 2024
topic: "moe-serving-systems"
status: "read"
priority: "novelty-boundary"
pdf: "PDFs/moe-serving-systems/2410.17043__aurora.pdf"
source_url: "https://arxiv.org/abs/2410.17043"
aliases:
  - "2410.17043"
  - "Aurora"
---

# Aurora

[PDF](PDFs/moe-serving-systems/2410.17043__aurora.pdf) · [Source](https://arxiv.org/abs/2410.17043)

> **Exploratory note for the candidate MoE-hetero-serving direction** (not part of the
> AutoPD-RL thesis). This is the paper that most directly *occupies the static core* of the
> spitball "place/colocate/route experts across heterogeneous GPUs." Read it as the
> boundary to carve against, not a neighbor to build on.

## TL;DR
Jointly optimizes **MoE expert placement + all-to-all communication scheduling** to minimize
inference time, analyzed across a 2×2 of {exclusive vs. colocated experts} × {homogeneous vs.
heterogeneous GPUs}. Proves optimal placement+schedule in 3 of 4 cases via exchange arguments;
shows **colocating experts on heterogeneous GPUs is NP-hard**, gives a poly-time bipartite-matching
approximation (1.07× optimality gap). 2.38× (homo) / **3.54× (hetero)** speedup on Google
production MoE traces. **Theory + simulation; no real hardware, no vLLM/SGLang integration.**

## Why This Matters (for the candidate direction)
- **Claims the static hetero-MoE placement cell.** "Hetero-aware MoE expert placement + comm
  scheduling" as a standalone thesis collides head-on with Aurora — and it already *proved* the
  hard case. Do **not** pitch static placement.
- **But it is offline/static, fully.** Optimizes from *"historical statistics… known a priori"*
  (§2.4). No expert migration, no replication-for-hotness, no online re-placement, no
  non-stationary-workload handling. Robustness test = up to 75% request noise applied *after*
  deployment (15.8% degradation), **not** continuous re-optimization.
- Its own gap list points straight at the surviving wedge: **"online dynamic rebalancing for
  temporal workload variation."** That sentence is the opening.

## Key Claims (verified, from fetched HTML)
- 2×2 cluster taxonomy; optimal in 3/4 (Thms 4–6), NP-hard for colocated-on-hetero (§7).
- Heterogeneity modeled = per-GPU compute (FLOPS) + bandwidth `B_i`; "big-switch" non-blocking
  network assumption (no oversubscription). At most 2 models/GPU.
- Speedups 2.38×/3.54×; 1.5× util vs. same-model colocation; 1.07× gap in NP-hard case.
- **Static/offline only** — confirmed verbatim. No runtime adaptation mechanism.

## Carve-out (if the pivot proceeds)
The defensible wedge is **what Aurora explicitly does not do**: *online, workload-adaptive*
placement/replication/migration under **non-stationary expert hotness**, on **real heterogeneous
hardware**, integrated with a production serving stack. Same structural bet as [[Research Plan]]'s
AutoPD-RL ("static optimum mis-fits a moving target → make it online"), applied to expert hotness.
**Premise to prove first:** expert hotness is non-stationary enough on a real serving stream to
defeat Aurora's static plan (else no wedge).

## Caveats
- Pre-cutoff (Oct 2024). Theory/sim — verify the "no real hardware" claim against the PDF before
  citing as the empirical-gap opening.

## Links
- [[MegaScale-Infer]] (production sibling; static placement + greedy hot-expert replication) ·
  [[HexAGenT]] (hetero scheduling, dense/agentic — not MoE) · [[Research Plan]]
