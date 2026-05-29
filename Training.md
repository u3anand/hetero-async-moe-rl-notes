# Training

Per step: forward → loss → backward → optimizer.

## Memory cost (BF16 weights/grads + FP32 AdamW)
- Parameter: 2 B
- Gradient: 2 B
- Optimizer state: 8 B (FP32 first/second moments + master weights; some setups use 12 B)
- Activations: depend on batch × seqlen, mitigated by activation recompute

→ ~**16 B/param floor** before activations. (E.g. a 10B-param model needs ~160 GB just for state.)

## Parallelism axes
- **DP** (data parallel) — replicate model, split batch. All-reduce gradients each step.
- **FSDP / ZeRO-3** — shard param/grad/optim across DP ranks. All-gather params per layer in fwd and bwd. Trades comms for memory.
- **TP** (tensor parallel) — split a matmul across GPUs within a layer. All-reduce activations per layer. Needs high-bandwidth intra-node link (NVLink-class).
- **PP** (pipeline parallel) — split layers across GPUs. Send activations across stages, microbatch to hide bubble.
- **EP** (expert parallel) — MoE-specific. Shard experts across ranks. **All-to-all** dispatch and combine per MoE layer in fwd and bwd. See [[MoE Architecture]].

Composable: e.g. FSDP × TP × PP × EP. Each axis layers its own collectives.

→ Deep Dives: [[Deep Dives/Dense Training Systems|Dense Training Systems]] (FSDP/TP/PP composition, recompute, FP8), [[Deep Dives/MoE Training Systems|MoE Training Systems]] (EP, all-to-all kernels, hetero MoE)

## Hetero cluster pain
Every collective syncs to its slowest member. A parallel group spanning fast + slow GPUs runs at slow speed.

Mitigation: pin each parallel group to a homogeneous tier; bridge tiers only via async or loose communication patterns (e.g. weight broadcast, trajectory upload — not per-step collectives).

→ MoE makes this worse: EP all-to-all spans every rank holding an expert, so it can't be pinned to one tier without breaking expert parallelism. See [[MoE vs Dense Workload]].
