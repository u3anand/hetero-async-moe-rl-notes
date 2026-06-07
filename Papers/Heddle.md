---
paper_id: "arxiv:2603.28101"
title: "Heddle: A Distributed Orchestration System for Agentic RL Rollout"
year: 2026
topic: "rl-systems"
status: "read"
priority: "novelty-boundary"
pdf: "PDFs/rl-systems/2603.28101__heddle.pdf"
source_url: "https://arxiv.org/abs/2603.28101"
aliases:
  - "2603.28101"
  - "Heddle"
---

# Heddle

[PDF](PDFs/rl-systems/2603.28101__heddle.pdf) · [Source](https://arxiv.org/abs/2603.28101)

> Peking University (Zili Zhang, Xin Jin et al.). **Once feared as the main novelty threat — verified NOT a threat to AutoPD-RL.** Post-cutoff (2026-03).

## TL;DR
**Trajectory-centric** rollout orchestration that optimizes the *when/where/how* of agentic rollout to tame long-tailed (tool-call-heavy) trajectories. Mechanisms: trainable **runtime predictor** (static prompt analysis + dynamic context), **progressive priority** (escalate long-tail trajectories), **DP-based trajectory placement** (segregate long from short), **tool-call-interval opportunistic migration**, and a **trajectory-adaptive resource manager** that tunes **model-parallelism degree per trajectory**. Up to 2.5× rollout throughput.

## Why This Matters (for AutoPD-RL)
- **The critical distinction:** Heddle's "adaptive resource manager" tunes **MP degree within a fixed worker pool** — it does **NOT** size prefill vs decode instance pools. The paper explicitly calls PD-disaggregation **"orthogonal"** and says Heddle "can be seamlessly integrated to provide intra-stage heterogeneity."
- → Treating Heddle as overlapping a *PD-pool-sizing* contribution is a **category error per the paper's own framing**. It does threaten generic *trajectory-scheduling* ideas (the direction we already retired), not AutoPD-RL.
- Reconnect point: its runtime predictor is prior art if AutoPD-RL ever predicts per-trajectory PD pressure — cite as the trajectory-prediction precedent, distinguish on the *control target* (pool ratio vs MP degree / priority).

## Key Claims (verified)
- Trajectory-centric scheduling; runtime prediction; progressive priority; DP placement; tool-call-aware migration; adaptive per-trajectory MP — all TRUE.
- PD pool sizing/routing — **out of scope, by explicit statement.**

## Caveats
- Post-cutoff (2026-03); descriptive claims verified from HTML, confirm before heavy citation.

## Links
- [[Direction Validation]] · contrast: [[RollPacker]] (batch-round granularity) · [[HexAGenT]] (PD routing)
