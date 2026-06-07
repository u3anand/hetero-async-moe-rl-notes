---
paper_id: "arxiv:2308.15030"
title: "SwapMoE: Serving Off-the-shelf MoE-based Large Language Models with Tunable Memory Budget"
year: 2024
topic: "moe-serving-systems"
status: "read"
priority: "perimeter"
pdf: "PDFs/moe-serving-systems/2308.15030__swapmoe.pdf"
source_url: "https://arxiv.org/abs/2308.15030"
aliases:
  - "2308.15030"
  - "SwapMoE"
---

# SwapMoE

[PDF](PDFs/moe-serving-systems/2308.15030__swapmoe.pdf) · [Source](https://arxiv.org/abs/2308.15030) · ACL'24

> **Perimeter / mechanism-adjacent for [[Research Plan 2]].** A second instance of the "dynamic
> resident working set of experts" idea — but it buys its memory savings with **approximation**
> (accuracy drop), on a **single device**. Shows the third escape from the bandwidth wall
> (migrate / replicate / **approximate**), and marks a line TierShift chooses *not* to cross.

## TL;DR
Keeps a small dynamic set of important experts ("**Virtual Experts**") resident in fast memory and
pages the rest, giving a **tunable memory budget**. Trades memory↔latency↔accuracy: e.g. 14.2 → 4.7
GiB and ~50% latency cut for a small Rouge-2 drop (−0.041) on summarization (Switch Transformer).

## Why This Matters (for TierShift)
- **The third lever beyond migrate/replicate: approximate.** SwapMoE sidesteps the
  weight-movement-bandwidth problem by **not fetching the true expert on the critical path** — it
  serves a "close enough" resident set and accepts an accuracy hit. This is a real design point, and
  the one TierShift **rules out**: [[Research Plan 2]] serves a *fixed checkpoint exactly* (no quality
  claim, no accuracy trade). Cite to position TierShift as *exact-routing* vs SwapMoE's *approximate*.
- **Single-device memory tiering**, not GPU-tier↔GPU-tier compute; **no hardware heterogeneity**. So
  it's adjacent (dynamic working set) but not a competitor on the hetero-serving axis.

## Key Claims (verified, from fetched abstract)
- Dynamic resident "Virtual Experts" set + tunable memory budget; approximate (accuracy drop).
- 14.2→4.7 GiB, ~50% latency cut, −0.041 Rouge-2. Switch Transformer / summarization.

## Caveats
- Approximation makes its results non-comparable to an exact-serving system; use as a *design-space*
  reference (the approximate corner), not a baseline.

## Links
- [[Toward-Efficient-MoE-Inference]] (exact reactive cache, the Layer-0 anchor) · [[Research Plan 2]]
