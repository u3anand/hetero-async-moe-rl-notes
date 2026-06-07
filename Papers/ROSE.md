---
paper_id: "arxiv:2605.06534"
title: "ROSE: Rollout On Serving GPUs via Cooperative Elasticity for Agentic RL"
year: 2026
topic: "rl-systems"
status: "read"
priority: "systems-core"
pdf: "PDFs/rl-systems/2605.06534__rose.pdf"
source_url: "https://arxiv.org/abs/2605.06534"
aliases:
  - "2605.06534"
  - "ROSE"
---

# ROSE

[PDF](PDFs/rl-systems/2605.06534__rose.pdf) · [Source](https://arxiv.org/abs/2605.06534)

> Alibaba/ROLL team (Wei Gao et al.), companion to [[RollArt]]. **Post-cutoff (2026-05) — verify directly.**

## TL;DR
Opportunistically repurposes **underutilized serving GPUs** to execute RL rollouts. Three components: (1) **SLO-safe co-serving executor** co-locates serving + rollout models on the same GPUs without breaking serving SLOs; (2) **cross-cluster weight transfer engine** with shard-aware routing; (3) **elastic rollout scheduler** that dynamically routes rollouts across dedicated + opportunistic serving GPUs.

## Why This Matters (for AutoPD-RL)
- Adjacent on *resource elasticity for rollout*, but the elasticity lever is **harvest-idle-serving-GPU + SLO-safe co-location**, not **prefill/decode pool sizing**. Different control knob → complementary, not overlapping.
- Useful as motivation that rollout resourcing is non-stationary and worth controlling online; cite alongside [[RollArt]] for the "rollout resource allocation is live and unsolved" framing.

## Key Claims (verified)
- Serving clusters leave substantial GPU compute + memory idle; ROSE safely harvests it.
- Elastic rollout scheduler dynamically routes across dedicated + opportunistic GPUs.
- **Not** about MoE expert placement or SWE-specific profiling.

## Caveats
- Same author team as [[RollArt]] / [[RollPacker]].
- 2026-05 arXiv id — confirm details from the PDF before citing precisely.

## Links
- [[Direction Validation]] · [[RollArt]] · [[Research Plan]]
