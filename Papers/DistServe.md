---
paper_id: "arxiv:2401.09670"
title: "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving"
year: 2024
topic: "llm-serving-systems"
status: "read"
priority: "prior-art-static"
pdf: "PDFs/llm-serving-systems/2401.09670__distserve.pdf"
source_url: "https://arxiv.org/abs/2401.09670"
aliases:
  - "2401.09670"
  - "DistServe"
---

# DistServe

[PDF](PDFs/llm-serving-systems/2401.09670__distserve.pdf) · [Source](https://arxiv.org/abs/2401.09670) · OSDI'24 · [code](https://github.com/LLMServe/DistServe)

> **Founding static PD-disaggregation work** — the ancestor AutoPD-RL must distinguish from.

## TL;DR
Disaggregates prefill and decode onto different GPUs to kill PD interference; co-optimizes resource allocation + per-phase parallelism given TTFT/TPOT SLOs (up to 7.4× request-rate gain). The PD allocation/parallelism is computed via an **offline placement search** for a fixed workload — **no online autoscaling / no PD re-balancing**.

## Why This Matters (for AutoPD-RL)
- Establishes the mechanism (PD split + per-phase parallelism) but assumes **static workload + static weights + SLO objective** — three assumptions broken by RL rollout (non-stationary multi-turn demand, weights updated every step, freshness objective).
- The "we automate what DistServe sizes offline, under RL dynamics" framing is core to the AutoPD-RL pitch.

## Key Claims (verified)
- PD disaggregation; offline placement search; SLO (TTFT/TPOT) objective; static.

## Caveats
- Serving, not RL. Static sizing.

## Links
- [[Direction Validation]] · [[Splitwise]] · [[TokenScale]] · [[Research Plan]]
