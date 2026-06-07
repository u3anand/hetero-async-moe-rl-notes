---
paper_id: "arxiv:2504.03871"
title: "HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs"
year: 2025
topic: "moe-serving-systems"
status: "read"
priority: "perimeter"
pdf: "PDFs/moe-serving-systems/2504.03871__hetermoe.pdf"
source_url: "https://arxiv.org/abs/2504.03871"
aliases:
  - "2504.03871"
  - "HeterMoE"
---

# HeterMoE

[PDF](PDFs/moe-serving-systems/2504.03871__hetermoe.pdf) · [Source](https://arxiv.org/abs/2504.03871)

> **Perimeter note (candidate MoE-hetero direction).** Owns "attention/expert disaggregation
> across GPU generations" — but for **training**, not serving. Defines the split that
> [[MegaScale-Infer]] later took to serving; cite to show that split is spoken-for.

## TL;DR
**Zebra Parallelism**: assign attention blocks to **newer** GPUs and expert modules to **older**
GPUs (newer GPUs win on attention; older are still fine for experts), overlapping micro-batches in
a zigzag to hide bubbles. **Asymmetric Expert Assignment** ("gather and squeeze") offloads select
experts to attention GPUs to fill idle time. Up to **2.3×** over existing MoE training systems,
1.4× over an optimally-balanced hetero baseline; holds 95% throughput with half an A40 cluster
swapped to V100.

## Why This Matters (perimeter)
- **Training only** — verbatim: *"we present HeterMoE to efficiently train MoE models with
  heterogeneous GPUs."* Inference/serving **absent from scope and future work**.
- **Static/offline** expert assignment (profile once → fixed per-layer assignment). No dynamic
  rebalancing.
- Take-away for the wedge: the attention/expert-split-across-tiers idea is **proven and claimed**
  (training by HeterMoE, serving by MegaScale-Infer). The candidate direction must **not** re-pitch
  that split — its novelty has to be the **online/dynamic** layer neither does.

## Heterogeneity & Results
A40/V100, L40S/T4 on-prem; simulated 200 Gbps AWS. ≤32K seq len.

## Links
- [[MegaScale-Infer]] (same split, serving) · [[Aurora]] · [[Research Plan]]
