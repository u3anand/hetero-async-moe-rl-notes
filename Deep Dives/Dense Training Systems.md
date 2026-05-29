# Dense Training Systems

Systems perspective on training dense LLMs. The four parallelism axes, mixed precision, recompute.

## Memory cost recap

See [[Training]] for the ~16 B/param floor (BF16 weights + grads + FP32 AdamW state). The job of a training system is to fit this state across GPUs and overlap the resulting communication with compute.

## DP / FSDP / ZeRO

- **DP (Data Parallel)** — full replica per GPU, split batch. All-reduce grads each step. Memory: full model per GPU.
- **ZeRO-1** — shard optimizer state across DP ranks. ~50% saving.
- **ZeRO-2** — shard optimizer state + grads. ~75% saving.
- **ZeRO-3 / FSDP** — shard everything (params + grads + optimizer). All-gather params per fwd/bwd layer; release after. ~94% saving.

Tradeoff: more sharding → more comms. FSDP has high collective volume (`O(P)` per fwd + `O(P)` per bwd in params alone), so it's bandwidth-sensitive. **PyTorch FSDP** is the production impl; **FSDP2** (current) does per-parameter sharding and composes cleanly with other axes.

## TP (Tensor Parallelism, Megatron-style)

Split a matmul column-wise or row-wise across `tp` GPUs.

- Linear: split weight matrix → each GPU computes a partial result → **all-reduce activations**.
- Attention: split heads across `tp` ranks; QKV + output projection sharded.
- FFN: gate/up sharded column-wise, down row-wise → fused into 2 all-reduces per transformer block.

Comms: **2 all-reduces per transformer block** (one in attention, one in FFN). Activations are large → needs intra-node bandwidth (NVLink, 600+ GB/s). Bad across nodes.

Rule of thumb: **TP within node only**.

## PP (Pipeline Parallelism)

Split layers across `pp` GPUs; microbatch the input so pipeline stages process different microbatches concurrently.

Schedules:
- **GPipe** — naive, fwd-all-then-bwd-all. Big bubble.
- **1F1B (PipeDream-Flush)** — interleave fwd and bwd. Smaller bubble.
- **Interleaved 1F1B** — multiple non-contiguous layer chunks per stage. Smaller bubble at the cost of more comms.
- **Zero-bubble PP (2024)** — split backward into "activation-grad" and "weight-grad" phases; reorder to fill the bubble. ~23% throughput gain.

Comms: send activations + grads across stages. Lower volume than TP, OK across nodes.

## Activation memory & recompute

Forward activations are stored for backward. At long context they dominate memory.

- **Full recompute** — store nothing, recompute every activation in bwd. Saves all activation memory, costs ~33% extra FLOPs.
- **Selective recompute** (Megatron, 2023) — store cheap activations (attention output projection), recompute expensive ones (attention softmax). Cuts ~70% of activation memory at ~3% FLOP overhead. SOTA.

## Mixed precision

- **BF16** — fwd/bwd in BF16, optimizer state in FP32. Standard since GPT-3.
- **FP8** — H100/B200 era. Weights + activations in FP8 (E4M3 or E5M2 depending on tensor role); optimizer in FP32. Per-tensor scaling required. ~2× throughput vs BF16. NVIDIA Transformer Engine + DeepSeek-V3 demonstrate FP8 training at scale.

## Composing the axes

Real training uses **3D or 4D parallelism**: DP × FSDP × TP × PP. Example layout for a 70B dense on 64× H100:
- TP = 8 (within node)
- PP = 2 (across two nodes)
- DP = 4 (across the remaining replicas)
- FSDP shards inside each DP rank

Each axis carries its own collective; the systems work is making them overlap with compute.

## Heterogeneous clusters

Every collective syncs to slowest. Pin parallel groups to a homogeneous tier (see [[Training]]). MoE breaks this composition because EP is global → see [[MoE Training Systems]].

## Sources

- *PyTorch FSDP* (VLDB 2023) — [arXiv:2304.11277](https://arxiv.org/abs/2304.11277)
- *Megatron-LM* (Shoeybi et al., 2019) — [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- *Reducing Activation Recomputation in Large Transformer Models* (Korthikanti et al., MLSys 2023) — [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)
- *Zero Bubble Pipeline Parallelism* (2024) — [arXiv:2401.10241](https://arxiv.org/abs/2401.10241)
- Book: Stas Bekman, *ML Engineering Open Book* — [github.com/stas00/ml-engineering](https://github.com/stas00/ml-engineering)
- Doc: Microsoft DeepSpeed ZeRO tutorial — [deepspeed.ai/tutorials/zero](https://www.deepspeed.ai/tutorials/zero/)
