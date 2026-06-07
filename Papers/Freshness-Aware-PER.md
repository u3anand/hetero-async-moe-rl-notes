---
paper_id: "arxiv:2604.16918"
title: "Freshness-Aware Prioritized Experience Replay for LLM/VLM Reinforcement Learning"
year: 2026
topic: "rl-systems"
status: "read"
priority: "objective-adjacent"
pdf: "PDFs/rl-systems/2604.16918__freshness-aware-per.pdf"
source_url: "https://arxiv.org/abs/2604.16918"
aliases:
  - "2604.16918"
  - "Freshness-Aware-PER"
---

# Freshness-Aware-PER

[PDF](PDFs/rl-systems/2604.16918__freshness-aware-per.pdf) · [Source](https://arxiv.org/abs/2604.16918)

> Listed in ROLL repo "notable work". Post-cutoff (2026-04).

## TL;DR
Augments PER priorities with a multiplicative exponential **age decay** to fight priority staleness as the policy evolves quickly. Purely **algorithmic** replay/sample-freshness — no GPU/PD resource scheduling.

## Why This Matters (for AutoPD-RL)
- Relevant to the **objective**, not the mechanism: it formalizes "fresher samples are worth more," which is the *training-side* counterpart of AutoPD-RL's "fresh complete trajectories per deadline" target (cf. [[RollArt]] SampleBuffer idle).
- Cite to motivate freshness as a first-class objective; distinguish clearly — they reweight replay, we allocate compute.

## Key Claims (verified)
- Exponential-age-decay priority reweighting; algorithmic, not resource scheduling.

## Caveats
- Don't conflate with resource scheduling. Post-cutoff — verify from PDF.

## Links
- [[Direction Validation]] · [[RollArt]]
