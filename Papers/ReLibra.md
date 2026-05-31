---
paper_id: "arxiv:2605.08639"
title: "ReLibra: Routing-Replay-Guided Load Balancing for MoE Training in Reinforcement Learning"
year: 2026
topic: "rl-systems"
status: "unread"
priority: "blocker"
pdf: ""
source_url: "https://arxiv.org/abs/2605.08639"
aliases:
  - "2605.08639"
  - "ReLibra"
---

# ReLibra

[Source](https://arxiv.org/abs/2605.08639)

**Why this is a blocker:** closest adjacency to [[Research Plan|the proposed direction]]. Need to confirm whether "hierarchical network bandwidth" in the abstract means homogeneous-with-NUMA or actually multi-tier hetero. If the latter, the asymmetric-expert-staleness wedge shrinks.

## What the abstract says

- Two load-balancing mechanisms at inter- and intra-batch timescales, matched to **hierarchical network bandwidths**.
- Inter-batch: expert reordering for cross-node balancing.
- Intra-batch: dynamic expert replication within node to absorb micro-batch load fluctuations.
- Exploits that rollout and training process identical tokens with identical MoE params → routing decisions known before training starts.
- Identifies hot-expert shift across micro-batches in RL → historical load prediction fails.

## Initial read (abstract-only, 2026-05-28)

- (1) Hetero clusters? — "hierarchical bandwidth" suggests tiered network but not explicitly heterogeneous compute.
- (2) Async replicas at different staleness? — not mentioned.
- (3) Phenomenon — load imbalance under sharp routing fluctuations, **not** cross-replica routing divergence.
- (4) Eval — appears homogeneous, baselines are Megatron-LM and EPLB.
- (5) Expert staleness as tier-speed function — not modeled.

Pending full-paper read.

## Systems Lens
- Workload: MoE RL training, rollout-aware load balancing.
- Bottleneck: hot-expert load imbalance with high temporal variance.
- Scheduling / placement: inter-batch expert reordering, intra-batch expert replication.
- Communication: hierarchical bandwidth-matched ops.
- Heterogeneous cluster angle: **TBD — gating question for [[Research Plan]].**
- Relevance to MoE RL / SWE-RL: directly adjacent.

## Key Claims
_TODO after full read_

## Caveats
_TODO_

## Links
- [[Research Plan]]
- [[Papers/R3|R3]]
- [[Deep Dives/MoE Load Balancing|MoE Load Balancing]]
