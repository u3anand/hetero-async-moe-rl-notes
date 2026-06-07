---
paper_id: "arxiv:2502.07903"
title: "HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environment"
year: 2025
topic: "hetero-serving-systems"
status: "read"
priority: "perimeter"
pdf: "PDFs/hetero-serving-systems/2502.07903__hexgen-2.pdf"
source_url: "https://arxiv.org/abs/2502.07903"
aliases:
  - "2502.07903"
  - "HexGen-2"
---

# HexGen-2

[PDF](PDFs/hetero-serving-systems/2502.07903__hexgen-2.pdf) · [Source](https://arxiv.org/abs/2502.07903) · ICLR'25

> **Perimeter note (candidate MoE-hetero direction).** "PD-disaggregated serving on heterogeneous
> GPUs" — the serving analogue closest to AutoPD-RL's substrate — **but dense LLMs only.** Cite to
> show hetero **PD disaggregation** is already done for dense serving; the MoE-expert axis is the gap.

## TL;DR
Disaggregates **prefill/decode across heterogeneous GPUs**; formalizes resource allocation + per-phase
parallelism + inter-phase KV-cache comm as a **constraint-optimization** problem via **graph
partitioning + max-flow**. Up to 2.0× (avg 1.3×) throughput, 1.5× lower latency, or comparable
performance at **30% lower cost** vs. homogeneous high-end deployment.

## Why This Matters (perimeter)
- **Dense only** — evaluated on OPT-30B and Llama-2-70B; **no MoE / expert-parallel** mention.
- Offline placement-style optimization (max-flow/graph-partition), like [[Helix]]; the online-vs-static
  question isn't resolved from the abstract — **verify from PDF** if it becomes load-bearing.
- Take-away: hetero **PD disaggregation** for serving is published (dense). Combined with [[Helix]],
  the dense-hetero-serving perimeter is closed. The only opening is **MoE expert
  placement/replication/migration**, and specifically the **online** version [[Aurora]] /
  [[MegaScale-Infer]] skip.

## Links
- [[Helix]] · [[Aurora]] · [[MegaScale-Infer]] · [[HexAGenT]] (agentic, dense) · [[Research Plan]]
