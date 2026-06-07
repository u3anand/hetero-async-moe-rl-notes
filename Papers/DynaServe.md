---
paper_id: "arxiv:2504.09285"
title: "DynaServe: Unified and Elastic Execution for Dynamic Disaggregated LLM Serving"
year: 2025
topic: "llm-serving-systems"
status: "skimmed"
priority: "prior-art-dynamic"
pdf: "PDFs/llm-serving-systems/2504.09285__dynaserve.pdf"
source_url: "https://arxiv.org/abs/2504.09285"
aliases:
  - "2504.09285"
  - "DynaServe"
---

# DynaServe

[PDF](PDFs/llm-serving-systems/2504.09285__dynaserve.pdf) · [Source](https://arxiv.org/abs/2504.09285)

## TL;DR
Unified elastic execution for disaggregated serving that tunes **batch size, prefill:decode token ratio, and decode context length per batch** — fine-grained dynamic PD control.

## Why This Matters (for AutoPD-RL)
- Closest serving-side analog to "adjust the PD ratio online." Strongest example that per-batch PD tuning is solved for serving. AutoPD-RL's contribution is **not** the per-batch tuning but doing it under RL's non-stationary multi-turn rollout with train-loop coupling.

## Key Claims (verified, abstract)
- Per-batch tuning of batch size, P:D token ratio, decode context length.

## Caveats
- Serving; static weights; SLO objective.

## Links
- [[Direction Validation]] · [[TokenScale]] · [[DOPD]] · [[Arrow]]
