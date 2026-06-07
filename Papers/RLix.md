---
paper_id: "github:rlops/rlix"
title: "RLix: RL job manager for multi-job GPU sharing"
year: 2025
topic: "rl-systems"
status: "read"
priority: "adjacent"
pdf: ""
source_url: "https://github.com/rlops/rlix"
aliases:
  - "RLix"
---

# RLix

[Source](https://github.com/rlops/rlix) — **GitHub project, not an arXiv paper.**

## TL;DR
RL job manager for **multi-job GPU sharing**: divides GPU capacity across concurrent RL jobs by remaining rollout work, treats rollout as lowest-priority/preemptable, and elastically expands/shrinks rollout workers (loading inference weights only while active). Inspired by ROLL's Partial Overlapping scheduling; cross-listed in the [[ROLL]] README.

## Why This Matters (for AutoPD-RL)
- Adjacent prior art on **elastic rollout-worker sizing**, but the axis is **multi-job GPU sharing**, not prefill/decode split within one job. Not a direct competitor — cite to show the "elastic rollout resourcing" neighborhood and carve out the PD-ratio axis.

## Key Claims (verified)
- Multi-job GPU sharing; rollout-as-preemptable; elastic rollout worker pool.

## Caveats
- No paper / no formal eval; GitHub project (~286★ at check). Corrects an earlier "unverifiable" guess — it *is* real, just not a paper.

## Links
- [[Direction Validation]] · [[ROLL]] · [[ROSE]]
