---
paper_id: "arxiv:2511.20982"
title: "DOPD: A Dynamic PD-Disaggregation Architecture"
year: 2025
topic: "llm-serving-systems"
status: "skimmed"
priority: "prior-art-dynamic"
pdf: "PDFs/llm-serving-systems/2511.20982__dopd.pdf"
source_url: "https://arxiv.org/abs/2511.20982"
aliases:
  - "2511.20982"
  - "DOPD"
---

# DOPD

[PDF](PDFs/llm-serving-systems/2511.20982__dopd.pdf) · [Source](https://arxiv.org/abs/2511.20982)

> Serving-side dynamic PD. Post-cutoff (2025-11).

## TL;DR
Adjusts prefill/decode instance allocation to an optimal P/D ratio from **real-time load monitoring** — i.e. dynamic PD-ratio control for serving.

## Why This Matters (for AutoPD-RL)
- Second data point (with [[TokenScale]]) that online PD-ratio adjustment is established on the serving side. Same takeaway: AutoPD-RL differentiates on RL dynamics + objective, not the autoscaling primitive.

## Key Claims (verified — title/abstract high confidence; internal algorithm medium confidence behind compressed PDF)
- Real-time-load-driven dynamic P/D ratio.

## Caveats
- Confirm exact mechanism from PDF before citing precisely. Serving, not RL.

## Links
- [[Direction Validation]] · [[TokenScale]] · [[Arrow]] · [[DynaServe]]
