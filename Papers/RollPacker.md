---
paper_id: "arxiv:2509.21009"
title: "RollPacker: Taming Long-Tail Rollouts for RL Post-Training with Tail Batching"
year: 2025
topic: "rl-systems"
status: "read"
priority: "systems-core"
pdf: "PDFs/rl-systems/2509.21009__rollpacker.pdf"
source_url: "https://arxiv.org/abs/2509.21009"
aliases:
  - "2509.21009"
  - "RollPacker"
---

# RollPacker

[PDF](PDFs/rl-systems/2509.21009__rollpacker.pdf) · [Source](https://arxiv.org/abs/2509.21009) · NSDI'26 · [code](https://github.com/Farrrrland/RollPacker)

> Alibaba/ROLL team (Wei Gao et al.), companion to [[RollArt]] / [[ROSE]].

## TL;DR
Mitigates long-tail rollouts in **synchronous** RL post-training via **tail batching**: consolidate prompts that lead to long-tail responses into a small subset of "long rounds," keep the majority of "short rounds" balanced — cutting GPU bubbles without sacrificing accuracy. Full system adds elastic rollout parallelism, dynamic reward scheduling, stream-based training. 2.03×–2.56× over veRL, up to 2.24× over RLHFuse (Qwen2.5, up to 128× H800).

## Why This Matters (for AutoPD-RL)
- Same long-tail problem AutoPD-RL profiles, but the lever is **batch-round scheduling** (reorder execution), not PD pool sizing — complementary.
- Granularity contrast worth citing: **RollPacker = batch/round** level; [[Heddle]] = **individual-trajectory** level; AutoPD-RL = **prefill/decode resource-pool** level.

## Key Claims (verified)
- Tail batching = herd long-tail prompts into few long rounds; keep short rounds short/balanced.
- Targets *synchronous* (on-policy, accuracy-preserving) RL specifically.

## Caveats
- Synchronous-RL framing; AutoPD-RL targets async rollout — note the regime difference.

## Links
- [[Direction Validation]] · [[RollArt]] · [[Heddle]]
