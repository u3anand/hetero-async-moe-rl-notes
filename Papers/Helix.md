---
paper_id: "arxiv:2406.01566"
title: "Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow"
year: 2024
topic: "hetero-serving-systems"
status: "read"
priority: "perimeter"
pdf: "PDFs/hetero-serving-systems/2406.01566__helix.pdf"
source_url: "https://arxiv.org/abs/2406.01566"
aliases:
  - "2406.01566"
  - "Helix"
---

# Helix

[PDF](PDFs/hetero-serving-systems/2406.01566__helix.pdf) · [Source](https://arxiv.org/abs/2406.01566) · ASPLOS'25

> **Perimeter note (candidate MoE-hetero direction).** The canonical "hetero-aware MILP/max-flow
> serving scheduler" — **but dense LLMs only, no MoE.** This is the paper the *retired*
> MoE-hetero-MILP thesis would have collided with; cite to prove the dense-hetero cell is full and
> the MoE-specific cell is the only opening.

## TL;DR
Formulates serving over heterogeneous GPUs+network as a **max-flow problem** on a weighted graph;
solves **MILP** to jointly optimize **model placement + request scheduling**. Up to 2.7× throughput
/ 2.8× lower prompt latency vs. Swarm / separate-pipeline baselines on a 42-node, 7-GPU-type cluster.

## Why This Matters (perimeter)
- **Dense only** — *"focuses exclusively on dense Transformer models (LLaMA)… no discussion of
  mixture-of-experts or dynamic expert routing."* No expert-parallel placement.
- **Static placement** — *"We only need to run model placement once for each cluster"* (MILP, up to
  4 h). Requests are then scheduled dynamically over the **fixed** partition; the placement itself
  does not adapt online.
- Take-away: "hetero serving via MILP/max-flow" is a **solved, published idea** for dense models. A
  MoE-hetero-serving pitch must add the **expert dimension** (placement/replication/migration) *and*
  the **online** axis — neither of which Helix touches.

## Heterogeneity & Results
A100 > L4 > T4 compute tiers; intra (10 Gb/s) vs inter-cluster (100 Mb/s, 50 ms) network. Static
known topology; no node churn / online re-optimization.

## Links
- [[HexGen-2]] (PD-disaggregated hetero, also dense) · [[Aurora]] (the MoE analogue) ·
  [[MegaScale-Infer]] · [[Research Plan]]
