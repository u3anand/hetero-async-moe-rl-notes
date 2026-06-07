---
paper_id: "neurips:2024/toward-efficient-moe-inference"
title: "Toward Efficient Inference for Mixture of Experts"
year: 2024
topic: "moe-serving-systems"
status: "read"
priority: "mechanism-anchor"
pdf: "PDFs/moe-serving-systems/neurips2024__toward-efficient-moe-inference.pdf"
source_url: "https://proceedings.neurips.cc/paper_files/paper/2024/hash/98bf3b8505c611ac21055dd9d355c66e-Abstract-Conference.html"
aliases:
  - "Toward Efficient Inference for Mixture of Experts"
  - "Expert Buffering"
---

# Toward-Efficient-MoE-Inference

[PDF](PDFs/moe-serving-systems/neurips2024__toward-efficient-moe-inference.pdf) · [Source](https://proceedings.neurips.cc/paper_files/paper/2024/hash/98bf3b8505c611ac21055dd9d355c66e-Abstract-Conference.html) · NeurIPS'24 · Meta (Huang, Ardalani, Bhosale, Ke, B. Lee, H-H. Lee, Sun, C-J. Wu) · [code](https://github.com/hyhuang00/moe_inference)

> **The mechanism anchor for [[Research Plan 2]] (TierShift) — Layer 0.** This is the paper that
> proves the *reactive cache* idea works for MoE experts, and resolves the "online prediction is
> weird" worry: it **does not predict**, it **caches** recently-hot experts. We build on its
> locality finding; we do not re-claim it.

## TL;DR
Characterizes two MoE inference workloads (LM, MT) and proposes three techniques: **Dynamic gating**,
**Expert Buffering**, and **Expert load balancing**. Expert Buffering *"exploits high temporal locality
across experts by allocating a fixed, but limited, amount of GPU memory for hot, active experts and
relying on CPU memory to buffer all other experts"* — i.e. a **reactive cache** keyed on recent
activation, not a workload forecast. Reduces static memory 1.47×; dynamic gating lifts throughput
6.2–11.6× (LM).

## Why This Matters (for TierShift)
- **Reactive, not predictive.** The hot set is kept resident because *recently-active experts are
  reused* (temporal locality) — the exact answer to "can we predict the workload?": you don't, you
  react. A cache is workload-agnostic, so it generalizes to anyone's traffic. This is **Layer 0** of
  the [[Research Plan 2]] novelty stack.
- **But it's a different problem than TierShift** — and this is the carve-out:
  - Expert Buffering is a **memory cache** (GPU↔CPU): the cold tier is *storage*, a miss is a **stall
    to fetch**. TierShift's cold tier is a **slow GPU that still computes**: a "miss" is not a stall,
    it's a **straggler** in the all-to-all. Objective flips from *hit-rate / memory* to
    *makespan / straggler across parallel compute tiers*.
  - No **replicate-and-split** (their CPU copy is backup, not a load-sharing replica). TierShift's
    token-splitting across tier replicas has no analog here.
  - No **hardware-tier gap** (one GPU + CPU, not fast-GPU vs slow-GPU compute).
- **Consequence:** the *naive port* of Expert Buffering to two GPUs (LRU hot→fast) is TierShift's
  **Layer-1 baseline**, not its contribution. The contribution (Layer 2) must beat it via the
  load-balancing + split levers this paper doesn't have.

## Key Claims (verified, secondary sources + abstract)
- 3 techniques: dynamic gating, Expert Buffering (cache), expert load balancing.
- Expert Buffering = reactive cache exploiting temporal locality; 1.47× static-mem reduction.
- Workloads: LM + MT (machine translation). Not heterogeneous-GPU; not compute-tier.

## Caveats
- PDF copies (proceedings.com / seas.upenn) failed clean text extraction — claims sourced from the
  NeurIPS abstract + secondary summaries; **re-verify locality/reuse numbers from the code repo or a
  clean PDF** before citing exact figures.

## Links
- [[SwapMoE]] (dynamic resident set, but approximate + single-device) · [[Research Plan 2]] (§3 Layered novelty) · [[Aurora]] · [[MegaScale-Infer]]
