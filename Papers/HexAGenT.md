---
paper_id: "arxiv:2605.16637"
title: "HexAGenT: Efficient Agentic LLM Serving via Workflow- and Heterogeneity-Aware Scheduling"
year: 2026
topic: "rl-systems"
status: "read"
priority: "novelty-boundary"
pdf: "PDFs/rl-systems/2605.16637__hexagent.pdf"
source_url: "https://arxiv.org/abs/2605.16637"
aliases:
  - "2605.16637"
  - "HexAGenT"
---

# HexAGenT

[PDF](PDFs/rl-systems/2605.16637__hexagent.pdf) · [Source](https://arxiv.org/abs/2605.16637)

> HKUST / WeBank / Wuhan U / Tsinghua (You Peng, Binhang Yuan et al.). **Closest prior art on PD *routing* for agentic workloads — verified adjacent-but-distinct.** Post-cutoff (2026-05).

## TL;DR
Schedules **online agentic LLM workflows** (multi-step DAGs revealed as parents finish) across a **heterogeneous, PD-disaggregated** inference cluster to meet **workflow-level (end-to-end) latency SLOs**. Jointly picks prefill placement, decode placement, and local queue priority via projected-finish urgency; accounts for KV capacity + cross-stage KV-transfer latency. A100/H100/H200; Llama-3.1-70B, Qwen3-235B-A22B; ShareGPT/BFCL-v3/LATS.

## Why This Matters (for AutoPD-RL)
Two load-bearing differences keep our wedge open:
1. **Serving, not RL rollout.** Objective = user-facing workflow latency SLO attainment. AutoPD-RL's objective = fresh complete trajectories per training deadline / rollout throughput, with continuously-updated weights and train/rollout contention.
2. **Routing over fixed pools, not sizing.** HexAGenT **assumes a pre-given P/D instance split** (e.g. "8 prefill + 8 decode") and only routes requests within it. Deciding/adapting the P:D count is exactly what it leaves untouched.

## Key Claims (verified, from fetched abstract + HTML)
- PD disaggregation across heterogeneous GPUs: **yes** (central).
- Automatic/dynamic P-vs-D pool sizing: **no** — pool sizes are input.
- Trajectory routing across prefill-heavy/decode-heavy hardware classes: only standard per-call P/D routing, no phase→hardware-class matching.
- Results: avg SLO-scale reduction 20.1% @95% attainment, 33.0% @99%.

## Caveats
- Cite as *the* closest scheduling/routing neighbor; carve out (i) fixed pools and (ii) serving-SLO objective. Post-cutoff — verify from PDF.

## Links
- [[Direction Validation]] · [[RollArt]] (RL-side, static) · [[Heddle]] (trajectory MP)
