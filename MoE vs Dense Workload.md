# MoE vs Dense Workload

Side-by-side along the axes that drive system design. The centerpiece note for understanding why MoE isn't just "a sparser dense model" from the systems perspective.

## Per-token compute
- **Dense:** FLOPs = `2 · P`. Every param participates.
- **MoE:** FLOPs = `2 · P_active`. Only top-k experts fire per token.
- → MoE's per-token cost looks like a small dense model. Cheap in FLOPs.

## Memory footprint
- **Dense:** hold `P` params (shardable via FSDP/TP).
- **MoE:** hold all `N` experts somewhere, even though only `k` fire per token. Footprint scales with `P_total`, not `P_active`.
- Mitigation: Expert Parallelism (EP) shards experts across ranks (see [[MoE Architecture]]).

## Communication

**Dense, per layer:**
- TP: 1 all-reduce on activations.
- FSDP: 1 all-gather on params (per fwd, per bwd).

**MoE adds, per MoE layer:**
- **All-to-all dispatch** — send each token to the `k` ranks that hold its experts.
- **All-to-all combine** — collect expert outputs back to the token's home rank.
- Both in fwd, both in bwd. Bandwidth-heavy, latency-sensitive, tail-bound on slow links.

## Routing dynamics
- Token-conditional → expert utilization is uneven in space (across experts) and time (across training).
- **Hot experts** bottleneck the EP step — the rank holding the hottest expert finishes last.
- **Cold experts** waste capacity, can drift or collapse.
- Aux loss helps; doesn't fix.

## Inference behavior
- **Dense:** predictable per-token cost. Clean batching.
- **MoE:** batch composition affects expert utilization. Same batch can saturate some experts and idle others. Batching efficiency depends on **expert co-occurrence** in the batch, not just sequence length.

## Training behavior
- **Dense:** every param sees a gradient every step.
- **MoE:** each expert is only updated by tokens routed to it → **effective batch per expert is smaller and variable**. Cold experts get noisy or no gradient. Drift is real.

## Hetero cluster impact (the key wedge)
- Dense parallelism (TP/PP/DP/FSDP) syncs to the slowest rank → painful on hetero, but addressable by pinning parallel groups to a homogeneous tier (see [[Training]]).
- MoE adds EP all-to-all, which spans **every rank holding an expert**. If experts are sharded across tiers, every token's roundtrip pays the slow-link cost. EP collapses to `slow-link bandwidth × per-layer all-to-all volume`.
- → "Hetero" + "MoE" interact non-trivially. Composing existing solutions (hetero RL scheduler + homogeneous-cluster MoE EP) does **not** give a working system — the slow tier strangles EP.

→ Deep Dive: [[MoE Training Systems|MoE Training Systems]] (EP all-to-all mechanics + hetero MoE pretraining systems)

## RL training implications
- **Rollouts**: per-token routing varies → cache reuse depends on expert co-location, not only prefix overlap (see [[Inference]]).
- **Async rollout replicas**: each replica has its own routing distribution → train/inference KL divergence grows; asymmetric expert staleness across replicas.
- **Sparse per-expert gradient** → updates per expert depend on routing distribution that itself drifted since the replica's weights were broadcast.

→ Deep Dive: [[MoE Inference Systems|MoE Inference Systems]] (RL-on-MoE rollout stacks, hot-expert replication)

These are the live open seams in the paper-direction search (see agent memory: `research-direction`).
