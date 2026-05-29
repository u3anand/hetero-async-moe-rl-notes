# MoE Training Systems

Systems perspective on training MoE models. The new piece on top of [[Dense Training Systems]] is **expert parallelism + all-to-all**.

## Expert Parallelism (EP)

The natural way to split MoE: each EP rank holds a subset of experts. When a token routes to expert `i`, the token's hidden state is sent to the rank holding expert `i`, computed, and the result returned.

Per MoE layer in fwd:
1. Router runs locally on all ranks.
2. **All-to-all dispatch** — every rank sends its tokens to the ranks holding the chosen experts.
3. Each rank runs its experts on the tokens it received.
4. **All-to-all combine** — results flow back to the originating ranks.
5. Weighted combine on the home rank.

Same shape in bwd. **Two all-to-alls per MoE layer per direction = 4 all-to-alls per layer per step.** The most expensive collective in modern LLMs.

## All-to-all volume and topology

Per-token data volume per all-to-all: `d_hidden · bytes`. At 8K tokens/step × 64 layers × 4 all-to-alls × 16K hidden × 2 B (BF16) ≈ **~67 GB of communication per step from MoE alone**.

All-to-all is sensitive to:
- **Topology** — NVLink within node (fast), NVSwitch full-bisection (fastest), PCIe/Ethernet across nodes (slow). EP across nodes is brutal.
- **Imbalance** — slowest expert holder dictates each all-to-all's completion.
- **Padding** — naive implementations pad to max-load-per-expert → wasted bandwidth.

## Kernel implementations

- **Naive (early DeepSpeed-MoE, FastMoE)** — pad-and-all-to-all, separate kernels for dispatch / GEMM / combine.
- **MegaBlocks (2023)** — reformulates MoE as **block-sparse matmul**; eliminates all-to-all padding via variable-shape sparse kernels. 2.4× over Megatron-MoE. Production-quality.
- **Tutel (2022)** — adaptive routing + custom CUDA all-to-all + fused dispatch/GEMM.
- **DeepEP** (DeepSeek) — production all-to-all kernels for V3-scale training.
- **FlashMoE (2025)** — fuses the entire MoE layer (dispatch + expert compute + combine) into a single GPU kernel. ~30–55% comms cost reduction.

## Composing EP with other parallelism (DeepSpeed-TED)

EP doesn't replace DP/FSDP/TP/PP — it composes.

- **EP × DP** — experts replicated across DP groups; gradients all-reduced across replicas.
- **EP × FSDP** — shard expert params across FSDP ranks within an EP group.
- **EP × TP** — split each expert's FFN across TP ranks. Useful for wide experts; adds another all-reduce per expert.
- **EP × PP** — different MoE layers on different pipeline stages.

DeepSpeed-TED (Tensor-Expert-Data) canonicalized the combinations. **4D parallelism** is standard for >100B MoE.

## Activation memory implications

MoE activations are sparse but the dispatch buffers are not. Each MoE layer requires:
- Dispatch buffer: `tokens × k × d_hidden` (each token replicates to its `k` experts).
- Expert input/output buffers per rank.

Higher `k` → larger buffer. Fine-grained experts → finer dispatch overhead. Selective recompute (see [[Dense Training Systems]]) on MoE layers usually drops the all-to-all buffers and recomputes.

## Heterogeneous MoE training (the open territory)

Three published systems for hetero MoE **pretraining** (no RL):
- **HeterMoE** (EuroSys 2025) — disaggregate attention (newer GPUs) from experts (older GPUs); 2.3× on V100/A40 mix.
- **Hexa-MoE** (2024) — expert-specific operators, in-place computation; 0.5–4.3× speedup on hetero hardware.
- **Lazarus** (one of the user's core papers — `~/research/papers/lazarus.pdf`) — elastic MoE training with adaptive expert replication; handles dynamic membership.

None handle RL: no rollout fleet, no async, no weight broadcast. The **MoE-on-hetero-cluster RL** combination is the residual seam (see agent memory: `research-direction`).

## Why this matters for the research direction

- EP all-to-all spans the whole expert plane → can't pin EP to a homogeneous tier without breaking it.
- Slow-tier links throttle every EP step → composing AReaL-Hex (hetero RL, dense) with slime (homogeneous MoE RL) does not work; the EP all-to-all collapses.
- Hot-expert replication shifts the bottleneck but adds gradient-merging complexity.

## Sources

- *MegaBlocks* (Gale et al., 2023) — [arXiv:2211.15841](https://arxiv.org/abs/2211.15841)
- *Tutel* (Hwang et al., 2022) — [arXiv:2206.03382](https://arxiv.org/abs/2206.03382)
- *DeepSpeed-TED* (2023) — [arXiv:2303.06318](https://arxiv.org/abs/2303.06318)
- *FlashMoE* (2025) — [arXiv:2506.04667](https://arxiv.org/abs/2506.04667)
- *HeterMoE* (Jin et al., 2025) — [arXiv:2504.03871](https://arxiv.org/abs/2504.03871)
- *Hexa-MoE* (2024) — [arXiv:2411.01288](https://arxiv.org/abs/2411.01288)
- Doc: DeepSpeed MoE tutorials — [deepspeed.ai/tutorials/mixture-of-experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/)
