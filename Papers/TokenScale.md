---
paper_id: "arxiv:2512.03416"
title: "TokenScale: Dynamic Prefill/Decode Autoscaling for LLM Serving"
year: 2025
topic: "llm-serving-systems"
status: "read"
priority: "prior-art-dynamic"
pdf: "PDFs/llm-serving-systems/2512.03416__tokenscale.pdf"
source_url: "https://arxiv.org/abs/2512.03416"
aliases:
  - "2512.03416"
  - "TokenScale"
---

# TokenScale

[PDF](PDFs/llm-serving-systems/2512.03416__tokenscale.pdf) · [Source](https://arxiv.org/abs/2512.03416)

> **The strongest "PD-autoscaling already exists" data point.** Serving-side. Post-cutoff (2025-12).

## TL;DR
True dynamic PD autoscaling for serving: monitors token arrival rate + per-stage **"token velocity"** to dynamically adjust prefiller/decoder instance counts; **"Convertible Decoders"** temporarily flip decode GPUs to prefill during bursts.

## Why This Matters (for AutoPD-RL)
- This is why "dynamically size the PD pools" is **not novel by itself** — the serving world already does it. AutoPD-RL's defensible novelty must be the **RL-specific dynamics** (updated-every-step weights, batch-synchronous long-tail, train/rollout contention, freshness-per-deadline objective), NOT the autoscaling mechanism.
- **Mechanism to borrow + adapt:** token-velocity signal and convertible-decoder trick — then show why they break unmodified in RL rollout.

## Key Claims (verified, title+abstract+HTML)
- Token-velocity-driven dynamic prefiller/decoder count; convertible decoders for bursts.

## Caveats
- Serving, static weights, SLO objective. Post-cutoff — verify from PDF.

## Links
- [[Direction Validation]] · [[DOPD]] · [[Arrow]] · [[DynaServe]] · [[DistServe]]
