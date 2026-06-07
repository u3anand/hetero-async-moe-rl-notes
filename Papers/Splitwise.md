---
paper_id: "arxiv:2311.18677"
title: "Splitwise: Efficient Generative LLM Inference Using Phase Splitting"
year: 2023
topic: "llm-serving-systems"
status: "read"
priority: "prior-art-static"
pdf: "PDFs/llm-serving-systems/2311.18677__splitwise.pdf"
source_url: "https://arxiv.org/abs/2311.18677"
aliases:
  - "2311.18677"
  - "Splitwise"
---

# Splitwise

[PDF](PDFs/llm-serving-systems/2311.18677__splitwise.pdf) · [Source](https://arxiv.org/abs/2311.18677) · ISCA'24 (Microsoft Research)

## TL;DR
Splits prompt (prefill) and token (decode) phases onto separate, phase-specialized machine pools. Pool sizes determined **statically offline** via an event-driven cluster simulator; a limited dynamic **"mixed pool"** can move machines in/out to reduce fragmentation/meet SLOs at high load — but the core prompt:token ratio is fixed by the pre-computed design.

## Why This Matters (for AutoPD-RL)
- Cite as **partially-dynamic** static prior art: it has a buffer-pool, but not true online PD-ratio control driven by workload signals. AutoPD-RL = full online PD sizing under RL dynamics.

## Key Claims (verified)
- Phase-split pools; offline sizing; limited dynamic mixed pool.

## Caveats
- Serving; "dynamic" is a fragmentation buffer, not workload-driven PD autoscaling.

## Links
- [[Direction Validation]] · [[DistServe]] · [[TokenScale]]
