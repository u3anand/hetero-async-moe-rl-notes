---
paper_id: "arxiv:2505.11916"
title: "Arrow: Adaptive Scheduling for PD-Disaggregated LLM Serving with Elastic Instance Pools"
year: 2025
topic: "llm-serving-systems"
status: "skimmed"
priority: "prior-art-dynamic"
pdf: "PDFs/llm-serving-systems/2505.11916__arrow.pdf"
source_url: "https://arxiv.org/abs/2505.11916"
aliases:
  - "2505.11916"
  - "Arrow"
---

# Arrow

[PDF](PDFs/llm-serving-systems/2505.11916__arrow.pdf) · [Source](https://arxiv.org/abs/2505.11916)

## TL;DR
Adaptive scheduling for PD-disaggregated serving using **stateless instances + elastic instance pools** — instances flex between prefill and decode roles as load shifts.

## Why This Matters (for AutoPD-RL)
- Another serving-side elastic-PD point. Reinforces that the *elastic role-flipping* idea is taken on the serving side; AutoPD-RL must own the RL-rollout regime + objective, and can cite Arrow/[[TokenScale]]/[[DynaServe]] as the mechanisms it adapts.

## Key Claims (verified, abstract)
- Stateless instances; elastic PD pools; adaptive scheduling.

## Caveats
- Serving, SLO objective. Skim-level confidence.

## Links
- [[Direction Validation]] · [[TokenScale]] · [[DOPD]] · [[DynaServe]]
