---
paper_id: "arxiv:2506.12417"
title: "HarMoEny: Efficient Multi-GPU Inference of MoE Models"
year: 2025
topic: "moe-serving-systems"
status: "read"
priority: "novelty-boundary"
pdf: "PDFs/moe-serving-systems/2506.12417__harmoeny.pdf"
source_url: "https://arxiv.org/abs/2506.12417"
aliases:
  - "2506.12417"
  - "HarMoEny"
---

# HarMoEny

[PDF](PDFs/moe-serving-systems/2506.12417__harmoeny.pdf) · [Source](https://arxiv.org/abs/2506.12417) · McGill

> **The closest *dynamic* neighbor for [[Research Plan 2]] (TierShift)** — it already does online token
> redistribution + async expert fetching to fight MoE load imbalance. Carve on **homogeneity** and the
> **cost/SLO objective**, and verify the homogeneity claim from the PDF before leaning on it.

## TL;DR
Attacks expert/GPU **load imbalance** in multi-GPU MoE inference with two online techniques:
**dynamic token redistribution** (move tokens off overloaded GPUs to underutilized ones) +
**asynchronous expert prefetching**. Under heavy imbalance: **+37–70% throughput, −34–41% TTFT, −84%
GPU idle** vs next-best baseline. Application-layer.

## The carve-out (why TierShift survives)
- **Homogeneous GPUs.** HarMoEny redistributes tokens to *under-utilized* GPUs — balancing **load**,
  treating GPUs as **equal-speed**. It has **no tier-speed model**: it sends tokens to whoever is idle,
  not to whoever is *faster*. TierShift splits a hot expert's tokens by **tier speed** (the fast tier
  gets proportionally more even at equal utilization) — GEM's straggler insight + online splitting.
  *(Verify from PDF that HarMoEny assumes homogeneous hardware — the abstract implies it via "underutilized GPUs.")*
- **Objective = throughput/TTFT**, not **p99-SLO + $/token on a mixed fleet**.
- **Token redistribution ≈ Layer 1.5.** HarMoEny's redistribution is conceptually adjacent to
  TierShift's replicate-and-split — **this is the closest dynamic prior art**, so the carve must be
  explicit: *homogeneous load-balancing* vs *heterogeneous speed-aware split under cost/SLO*. Use it as
  a **baseline** (HarMoEny/PROBE-style dynamic token assignment), not just a citation.

## Key Claims (verified, secondary sources + abstract)
- Dynamic token redistribution + async expert prefetch; +37–70% tput, −34–41% TTFT, −84% idle.
- Targets skewed workloads / load imbalance; multi-GPU; application-layer.

## Caveats
- Re-verify the homogeneous-hardware assumption and whether any tier-speed term exists, from the PDF —
  it's the load-bearing distinction.

## Links
- [[GEM]] (static hetero mapping) · CRAFT / PROBE (homogeneous replication/prefetch) · [[Research Plan 2]]
