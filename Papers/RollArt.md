---
paper_id: "arxiv:2512.22560"
title: "RollArt: Scaling Agentic RL Training via Disaggregated Infrastructure"
year: 2025
topic: "rl-systems"
status: "read"
priority: "baseline-to-beat"
pdf: "PDFs/rl-systems/2512.22560__rollart.pdf"
source_url: "https://arxiv.org/abs/2512.22560"
aliases:
  - "2512.22560"
  - "RollArt"
---

# RollArt

[PDF](PDFs/rl-systems/2512.22560__rollart.pdf) · [Source](https://arxiv.org/abs/2512.22560)

> **This is the baseline-to-beat and the paper that *names our gap*.** Alibaba/ROLL team (Wei Gao et al.). Verified against primary source 2026-06-03.

## TL;DR
Scales agentic RL training by treating it as a heterogeneous workload and **disaggregating** it across best-fit hardware. Three principles: (1) **hardware-affinity workload mapping** — route compute-bound (prefill) vs bandwidth-bound (decode) vs CPU-heavy (env) work to best-fit GPU/CPU; (2) **trajectory-level asynchrony** — manage execution at the trajectory granularity to mitigate resource bubbles; (3) **statefulness-aware computation** — offload stateless components (e.g. reward models) to serverless. Plus async cross-cluster weight update (Mooncake-based).

## Why This Matters (for AutoPD-RL)
- It occupies the *exact* setting — heterogeneous-GPU prefill/decode for agentic RL — but **statically/manually**. PD affinity is set by decorators like `@rdist.hw_mapping(hw_affinity={"FrozenLake":"H800","default":"H20"})`.
- **It explicitly leaves our problem as future work** (verbatim): *"Many systems disaggregate prefill and decoding across heterogeneous GPUs … However, real deployments currently require manual configuration of prefill and decoding instances, which easily leads to load imbalance. We hence leave it as future work."* → This is the wedge for [[Direction Validation]] / AutoPD-RL.
- Evaluates **SWE-bench (prefill-heavy)** + **GEM-math (decode-heavy)** + GEM-game, FrozenLake, WebShop; Qwen3 8B–32B. So the "SWE workloads are missing" angle is dead; the *automatic PD sizing* angle is alive.

## Key Claims (verified)
- Agentic RL is heterogeneous: compute-intensive prefill + bandwidth-bound decode + stateful CPU-heavy env sims + reward + training + cross-cluster weight updates.
- **SampleBuffer bottleneck:** trainer's blocking `get_batch` waits for enough complete trajectories to hit batch size — *"consumes up to 62% of iteration time with GPU idleness."* ← the freshness/idle objective AutoPD-RL optimizes against.

## Caveats
- Companion work to [[ROSE]] and [[RollPacker]] (same author team) — treat as one group's program, not independent corroboration.
- Heterogeneity is at the phase/workload/infra level, not MoE-internals.

## Links
- [[Direction Validation]] · [[Research Plan]] · related: [[HexAGenT]] [[Heddle]] [[ROSE]] [[RollPacker]]
