# Attention Variants

Variants of multi-head attention, organized by what they optimize.

## Query/Key/Value head sharing — for cheaper KV cache

Standard MHA has `h` query heads and `h` matched K/V heads. KV cache size scales linearly with K/V head count. Variants reduce K/V heads while keeping query heads:

- **MHA** — `h` Q heads, `h` K heads, `h` V heads. Original Transformer.
- **MQA** (Multi-Query) — `h` Q heads, 1 K head, 1 V head. Smallest KV cache. Quality regression on bigger models.
- **GQA** (Grouped-Query) — `h` Q heads, `g` K heads, `g` V heads with `1 < g < h`. Each group of `h/g` Q heads shares one K/V. Sweet spot. Used in LLaMA 2/3, Mistral, Qwen.
- **MLA** (Multi-Head Latent Attention, DeepSeek-V2) — low-rank factorization of K and V into a small latent cache (`d_kv_lora` dims) that's projected up at attention time. Cache *smaller than MQA* with quality close to MHA. Used in DeepSeek-V2/V3.

KV-cache size relationship: `MHA > GQA > MQA`; MLA is orthogonal — it shrinks per-token cache via low-rank, not head count.

## FlashAttention — IO-aware exact attention

Standard attention reads/writes `O(N²)` to HBM for the `softmax(QKᵀ/√d)V` matrix. FlashAttention reorders the computation to avoid materializing `N×N` in HBM.

- **FlashAttention v1 (2022)** — tile `Q`, `K`, `V` into blocks; compute per-block in SRAM; online softmax avoids storing intermediates. 2–4× over naive.
- **FlashAttention v2 (2023)** — better work partitioning across thread blocks; fewer non-matmul FLOPs. ~2× over v1 at long context.
- **FlashAttention v3 (2024, H100)** — async warp specialization (TMA + WGMMA), FP8 support. The current SOTA for FP8 inference and training pipelines.

All three are **exact** (not approximate). The win is HBM bandwidth; the math is unchanged.

## Why this matters for the research direction

(See agent memory: `research-direction`.)

- **KV cache size** is the dominant memory cost in long-context inference (see [[Inference]]). GQA/MLA reduce this 4–8×.
- SWE-RL rollouts have long prompts (repo prefill) → KV cache pressure is acute.
- MLA's tiny KV cache is part of why DeepSeek-V3 (671B-A37B) is cheap to serve.
- FlashAttention-3 + FP8 is the current SOTA inference attention kernel on H100.

## Sources

- *FlashAttention* (Dao et al., 2022) — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- *FlashAttention-2* (Dao, 2023) — [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- *FlashAttention-3* (Shah et al., 2024) — [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- *GQA* (Ainslie et al., 2023) — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- *DeepSeek-V2* (2024, for MLA) — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- Blog: Lilian Weng, *The Transformer Family v2* — [lilianweng.github.io](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/)
