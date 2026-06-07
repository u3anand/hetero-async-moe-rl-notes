---
paper_id: "arxiv:2506.06122"
title: "ROLL: Reinforcement Learning Optimization for Large-Scale Learning"
year: 2025
topic: "rl-systems"
status: "read"
priority: "infra-base"
pdf: "PDFs/rl-systems/2506.06122__roll.pdf"
source_url: "https://arxiv.org/abs/2506.06122"
aliases:
  - "2506.06122"
  - "ROLL"
---

# ROLL

[PDF](PDFs/rl-systems/2506.06122__roll.pdf) · [Source](https://arxiv.org/abs/2506.06122) · [code](https://github.com/alibaba/ROLL)

## TL;DR
Alibaba's base RL scaling library and the ecosystem [[RollArt]] / [[ROSE]] / [[RollPacker]] build on. Ray-based multi-role distributed architecture for flexible resource allocation + heterogeneous task scheduling.

## Why This Matters (for AutoPD-RL)
- It's the substrate the wedge's neighbors sit on; understanding `AutoDeviceMapping` + async rollout is needed to position AutoPD-RL as the *automatic* layer above ROLL's manual device mapping.
- Possible implementation target / comparison harness alongside prime-rl.

## Key Claims (verified from README + paper)
- **AutoDeviceMapping**: custom role→device mapping for colocated/disaggregated deployments (the *manual* knob AutoPD-RL would automate).
- Sample-level async parallel rollout; async training.
- vLLM/SGLang inference; DeepSpeed (ZeRO)/Megatron 5D/FSDP (in progress) backends.
- Qwen3 (8/14/32B) + Qwen3-MoE (30A3/235A22) support.

## Caveats
- FSDP support listed as in-progress at paper time.

## Links
- [[Direction Validation]] · [[RollArt]] · [[RLix]]
