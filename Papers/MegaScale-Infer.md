---
paper_id: "arxiv:2504.02263"
title: "MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism"
year: 2025
topic: "moe-serving-systems"
status: "read"
priority: "novelty-boundary"
pdf: "PDFs/moe-serving-systems/2504.02263__megascale-infer.pdf"
source_url: "https://arxiv.org/abs/2504.02263"
aliases:
  - "2504.02263"
  - "MegaScale-Infer"
---

# MegaScale-Infer

[PDF](PDFs/moe-serving-systems/2504.02263__megascale-infer.pdf) · [Source](https://arxiv.org/abs/2504.02263) · ByteDance

> **Exploratory note for the candidate MoE-hetero-serving direction.** The production
> heavyweight in this space — the one to *respect and not compete with on scale*. It already
> owns "disaggregate attention/expert across heterogeneous tiers for MoE serving."

## TL;DR
**Disaggregated expert parallelism**: split **attention modules** and **expert/FFN modules** onto
separate GPUs (attention replicated for batch-building; experts consolidated to lift per-expert
batch size, turning FFN from memory-bound back to compute-bound). Explicitly runs the two roles on
**different GPU tiers** (H20 for attention = memory/BW-rich; L40S for experts = compute-rich) →
**1.7× throughput/cost**. ByteDance production, **~10k-GPU** deployment, 1.5–2.0× cost reduction.
Custom **M2N** comm library (4.2× over NCCL).

## Why This Matters (for the candidate direction)
- **Takes the "attention/expert disaggregation across hetero tiers" idea for *serving*.**
  ([[HeterMoE]] did the same split for *training*.) → **Do not propose that split as novelty.**
- **Still static placement at heart.** Deployment plan = **offline** search (Algorithm 1). Load
  imbalance handled by **"on-device redundancy based on expert popularity"** — a *greedy,
  retrospective* hot-expert replication from **historical** popularity, **not** real-time dynamic
  re-placement. Their own production data (Fig 16) shows heavy, persistent expert imbalance they
  admit they patch with "greedy heuristics on historical data."
- So even the strongest production system **punts on online adaptation** — same gap as [[Aurora]].

## Key Claims (verified, from fetched HTML)
- Heterogeneous setup is explicit and central: *"MegaScale-Infer supports a heterogeneous hardware
  setup for attention nodes and expert nodes."*
- Objective = max decoding throughput / unit cost under a 150 ms TBT SLO.
- vs. vLLM/TensorRT-LLM: 1.90× per-GPU decode throughput (homo, Ampere); 1.86× tput/cost (hetero
  H20+L40S). M2N: 4.2× tput, 68.2% latency cut vs. NCCL.
- Replication is **greedy on historical popularity**; no online migration.

## Carve-out (if the pivot proceeds)
Compete on the **dynamics axis, not scale/throughput** — you will lose at 10k GPUs. The opening is
**online adaptation to non-stationary expert hotness** (migration + re-replication as the workload
mix shifts), demonstrated at watgpu308 tier scale, *motivated by* MegaScale-Infer's own admission
that production expert load is imbalanced and only greedily patched offline.

## Caveats
- Production paper; "static + greedy replication" reading should be re-verified against §placement
  and §load-balancing in the PDF before it anchors the novelty claim.

## Links
- [[Aurora]] (static-theory sibling) · [[HeterMoE]] (same split, training-side) ·
  [[HexAGenT]] (hetero scheduling, dense/agentic) · [[Research Plan]]
