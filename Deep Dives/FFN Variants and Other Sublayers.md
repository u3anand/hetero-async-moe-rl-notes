# FFN Variants and Other Sublayers

Small architectural choices modern LLMs converged on. Grouped because each on its own is too small for a dedicated page.

## FFN — GLU variants

Original Transformer FFN: `FFN(x) = down(act(up(x)))` with `act = ReLU`. Modern variants use a **gated linear unit**:

```
FFN_glu(x) = down( act(gate(x)) ⊙ up(x) )
```

Three projections (`gate`, `up`, `down`) instead of two. The element-wise gate gives the FFN a multiplicative degree of freedom that improves quality.

- **SwiGLU** — `act = SiLU`. Used in LLaMA, PaLM, Qwen, Mistral.
- **GeGLU** — `act = GeLU`. Some Google models.
- **ReGLU** — `act = ReLU`. Less common.

To keep param count similar at iso-FLOPs, GLU FFNs use `d_ff ≈ (8/3)·d` instead of the classic `4·d` (3 projections instead of 2). Empirical: SwiGLU ~1–2% better perplexity at iso-compute. Cheap win.

## Norms — RMSNorm

LayerNorm: `(x - μ)/σ · γ + β` — re-centers and re-scales.

**RMSNorm** drops the mean re-centering, keeps the scale:
```
RMSNorm(x) = x / sqrt(mean(x²) + ε) · γ
```

No β, no mean. ~7% faster kernel, equivalent quality. Standard in modern open models (LLaMA, Qwen, DeepSeek).

## Pre-norm vs post-norm

- **Post-norm** (original Transformer): `x_out = LN(x + Sublayer(x))`. Norm after the residual.
- **Pre-norm** (modern): `x_out = x + Sublayer(LN(x))`. Norm inside the residual.

Pre-norm trains stably at depth without warmup tricks. Every modern decoder LLM is pre-norm.

## Positional encoding

Attention is permutation-equivariant by default — needs explicit position signal.

- **RoPE** (Rotary Position Embedding) — rotate Q and K vectors in 2D subspace pairs by an angle proportional to position. Implements relative positions implicitly; no positional embedding table. Standard in LLaMA, Qwen, Mistral, DeepSeek.
- **ALiBi** — additive linear bias to attention scores based on token distance. Cheap; extrapolates beyond training context length. Used in BLOOM, MPT.
- **Learned absolute positional embeddings** (GPT-2 style) — table indexed by position. Doesn't extrapolate. Mostly deprecated.

Long-context recipes often involve **RoPE scaling** (interpolating or extrapolating rotation frequencies — NTK-aware, YaRN).

## Tokenization (briefly)

Modern LLMs use BPE or its variants (BBPE, SentencePiece). Tokenizer size 32K–256K. For systems work the relevant fact is that token count varies across tokenizers for the same string — affects KV cache size and rollout token budgets.

## Sources

- *GLU Variants Improve Transformer* (Shazeer, 2020) — [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- *RMSNorm* (Zhang, Sennrich, NeurIPS 2019) — [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- *RoFormer / RoPE* (Su et al., 2021) — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- *ALiBi* (Press, Smith, Lewis, ICLR 2022) — [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
