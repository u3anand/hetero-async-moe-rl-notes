---
paper_id: "arxiv:2510.05943"
title: "EARL: Efficient Agentic RL Post-Training for LLMs under Dynamic Context Lengths"
year: 2025
topic: "rl-systems"
status: "read"
priority: "systems-core"
pdf: "PDFs/rl-systems/2510.05943__earl.pdf"
source_url: "https://arxiv.org/abs/2510.05943"
aliases:
  - "2510.05943"
  - "EARL"
---

# EARL

[PDF](PDFs/rl-systems/2510.05943__earl.pdf) · [Source](https://arxiv.org/abs/2510.05943) · EuroMLSys'26

## TL;DR
Targets the **dynamic / long context-length** bottlenecks of agentic RL: (1) context growth → memory/latency/OOM; (2) intermediate-tensor accumulation → cross-device data-movement bottleneck. Two mechanisms: a **parallelism selector** that adapts model + training parallelism across RL stages by sequence length and system load, and a **layout-aware data dispatcher** that replaces all-gather-and-scatter with all-to-all decentralized exchange.

## Why This Matters (for AutoPD-RL)
- Adjacent lever: **dynamic parallelism by sequence length**, distinct from PD pool sizing — but the "context length drives resource demand" signal is directly relevant to predicting prefill (long-context ingest) vs decode pressure in AutoPD-RL profiling.
- Cite as the *dynamic-parallelism-for-context* neighbor; distinguish on knob (parallelism layout vs PD pool ratio) and objective.

## Key Claims (verified)
- Sequence-length- and load-aware dynamic parallelism selection — TRUE.
- Layout-aware all-to-all decentralized dispatch — TRUE.

## Caveats
- Scope is long/dynamic context (memory + data movement), not long-tail latency scheduling.

## Links
- [[Direction Validation]] · [[Heddle]] · [[RollPacker]]
