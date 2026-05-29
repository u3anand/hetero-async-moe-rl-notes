---
paper_id: "arxiv:2510.23027"
title: "Towards Stable and Effective Reinforcement Learning for Mixture-of-Experts"
year: 2025
topic: "rl-systems"
status: "unread"
priority: "core"
pdf: ""
source_url: "https://arxiv.org/abs/2510.23027"
aliases:
  - "2510.23027"
  - "Router-Aware IS"
  - "RoIS"
---

# Router-Aware Importance Sampling

[Source](https://arxiv.org/abs/2510.23027)

Router-logit-guided rescaling of importance sampling weights for off-policy MoE RL.

## Core mechanism

Uses router logits to rescale IS weights so the IS correction reflects routing dynamics, not just per-token logprob ratios. Reduces gradient variance and mitigates training divergence in MoE off-policy RL.

## Why it matters here

- Confirms standard per-token IS is insufficient for MoE off-policy.
- Adjacent to (but distinct from) the cross-replica IS correction proposed in [[Paper Direction]] mechanism (2).

## What it does NOT address

- Multiple async replicas at tier-asymmetric staleness.
- Heterogeneous clusters.
- Cross-replica routing reconciliation.

## Systems Lens
- Workload: off-policy MoE RL.
- Bottleneck: gradient variance from naive IS.
- Scheduling / placement: N/A (algorithm-level).
- Heterogeneous cluster angle: none.

## Links
- [[Paper Direction]]
- [[Papers/R3|R3]]
