# Transformer

Decoder-only LLM = embeddings + L identical **blocks** + LM head.

## Block (pre-norm)
1. **Attention** — multi-head, causal-masked. Params per block: `4·d²` (Q, K, V, O projections). → Deep Dive: [[Deep Dives/Attention Variants|Attention Variants]]
2. **FFN** — 2-layer MLP. Modern variant SwiGLU: `down(silu(gate(x)) * up(x))`. Params per block ≈ `8·d²` (with d_ff ≈ 4d for classic; ~8/3·d for SwiGLU at iso-FLOPs). → Deep Dive: [[Deep Dives/FFN Variants and Other Sublayers|FFN Variants and Other Sublayers]]
3. Residual + RMSNorm around each sublayer.

## Where parameters live
- **FFN ≫ Attention ≫ Embeddings** in param count.
- FFN is the fat layer → it's the one MoE attacks (see [[MoE Architecture]]).

## Where compute lives
- Attention: `O(N²·d)` per layer in seqlen.
- FFN: `O(N·d²)` per layer.
- Long context → attention dominates; short context → FFN dominates.

## Hyperparams
`d` (hidden), `L` (layers), `h` (query heads), `n_kv_heads` (key/value heads, < `h` for GQA/MQA), `d_ff` (FFN intermediate), `vocab`.

## Reference code (Kimi K2 / DeepSeek-V3 structure)

Annotated reference. Structure follows Kimi K2 and DeepSeek-V3 open releases. Production code adds config plumbing, quantization, KV-slot management, FlashAttention dispatch, etc. — stripped here to keep shape and role legible. Upstream:

- Kimi K2: [github.com/MoonshotAI/Kimi-K2](https://github.com/MoonshotAI/Kimi-K2) — `modeling_deepseek.py` is the reference HF file (Kimi K2 inherits the DeepSeek-V3 architecture).
- DeepSeek-V3: [github.com/deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) — `inference/model.py`.

**Shape symbols used below**

| symbol | meaning | DeepSeek-V3 value |
|---|---|---|
| `B` | batch | — |
| `T` | sequence length this step (training: full; decode: 1) | — |
| `T_kv` | total KV positions (T at training; cumulative during decode) | — |
| `d` | model hidden dim | 7168 |
| `V` | vocab size | 129280 |
| `L` | number of decoder layers | 61 |
| `n_h` | query heads | 128 |
| `q_lora` | Q low-rank dim | 1536 |
| `kv_lora` | KV latent dim (the part that gets cached) | 512 |
| `qk_nope`, `qk_rope` | per-head Q/K dims (RoPE vs non-RoPE split) | 128, 64 |
| `v_head` | per-head V dim | 128 |
| `d_ff` | FFN intermediate (dense layers) or per-expert (MoE) | 18432 (dense) |

Per-token KV cache with MLA at this config: `kv_lora + qk_rope = 576` floats. GQA-equivalent quality would need `~2 · n_kv_heads · v_head ≈ 2048`. That ~3.5× reduction is one of the things that makes serving DeepSeek-V3 / Kimi K2 affordable.

```python
# === Sublayer norms / activations =========================================

class RMSNorm(nn.Module):
    """Role: scale-only normalization (no mean re-centering). Used twice per block, plus once before LM head."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))                  # learned scale: [d]
        self.eps = eps

    def forward(self, x):
        # x: [..., d]
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()  # [..., 1]
        return (x * rms) * self.weight
        # → [..., d]


class SwiGLU(nn.Module):
    """Role: per-token nonlinear mixing (the FFN). Three projections — gate, up, down.
    In MoE models this same module is the per-expert FFN, just with a smaller d_ff (see [[MoE Architecture]])."""
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)                 # weight: [d_ff, d]
        self.up   = nn.Linear(d, d_ff, bias=False)                 # weight: [d_ff, d]
        self.down = nn.Linear(d_ff, d, bias=False)                 # weight: [d, d_ff]

    def forward(self, x):
        # x: [B, T, d]   (or [N, d] when called per-expert with flattened tokens)
        return self.down(F.silu(self.gate(x)) * self.up(x))
        # gate(x), up(x): [B, T, d_ff]
        # product:        [B, T, d_ff]
        # → [B, T, d]


# === Attention ============================================================

class MLA(nn.Module):
    """Role: multi-head latent attention.
    Caches a small *latent* (kv_lora dims) plus a shared RoPE key (qk_rope dims) per token.
    Up-projects the latent to per-head K/V at attention time. Cache ~3-4× smaller than GQA at matched quality."""
    def __init__(self, d, n_h, q_lora, kv_lora, qk_nope, qk_rope, v_head):
        super().__init__()
        self.n_h = n_h
        self.qk_nope, self.qk_rope = qk_nope, qk_rope
        self.qk_head = qk_nope + qk_rope                           # per-head Q/K dim
        self.v_head  = v_head
        self.kv_lora = kv_lora

        # Q path: down → norm → up
        self.wq_a   = nn.Linear(d, q_lora, bias=False)             # [q_lora, d]
        self.q_norm = RMSNorm(q_lora)
        self.wq_b   = nn.Linear(q_lora, n_h * self.qk_head, bias=False)  # [n_h*qk_head, q_lora]

        # KV path: down to (latent || RoPE-K); latent gets norm'd + cached; up-proj built lazily.
        self.wkv_a   = nn.Linear(d, kv_lora + qk_rope, bias=False) # [kv_lora+qk_rope, d]
        self.kv_norm = RMSNorm(kv_lora)
        self.wkv_b   = nn.Linear(kv_lora, n_h * (qk_nope + v_head), bias=False)  # [n_h*(qk_nope+v_head), kv_lora]

        self.wo = nn.Linear(n_h * v_head, d, bias=False)           # [d, n_h*v_head]

    def forward(self, x, freqs_cis, attn_mask, kv_cache=None):
        # x: [B, T, d]
        B, T, _ = x.shape

        # --- Query: [B, T, d] → [B, T, n_h, qk_head]
        q = self.wq_b(self.q_norm(self.wq_a(x)))                   # [B, T, n_h*qk_head]
        q = q.view(B, T, self.n_h, self.qk_head)
        q_nope, q_rope = q.split([self.qk_nope, self.qk_rope], dim=-1)   # [B,T,n_h,qk_nope], [B,T,n_h,qk_rope]
        q_rope = apply_rope(q_rope, freqs_cis)

        # --- KV: [B, T, d] → cached latent [B, T, kv_lora] + shared RoPE key [B, T, 1, qk_rope]
        kv = self.wkv_a(x)                                          # [B, T, kv_lora + qk_rope]
        kv_latent, k_rope = kv.split([self.kv_lora, self.qk_rope], dim=-1)
        kv_latent = self.kv_norm(kv_latent)                         # [B, T, kv_lora]   ← CACHED
        k_rope    = apply_rope(k_rope.unsqueeze(2), freqs_cis)      # [B, T, 1, qk_rope] ← CACHED (shared across heads)

        if kv_cache is not None:
            # Append-only growth during decode. Returned shapes: [B, T_kv, kv_lora], [B, T_kv, 1, qk_rope]
            kv_latent, k_rope = kv_cache.append(kv_latent, k_rope)

        # Up-project the latent to per-head K (nope part) and V — done every step, not cached.
        kv     = self.wkv_b(kv_latent)                              # [B, T_kv, n_h*(qk_nope+v_head)]
        kv     = kv.view(B, -1, self.n_h, self.qk_nope + self.v_head)
        k_nope, v = kv.split([self.qk_nope, self.v_head], dim=-1)   # [B,T_kv,n_h,qk_nope], [B,T_kv,n_h,v_head]

        # Final K = [nope || rope]; nope per-head, rope broadcast.
        k      = torch.cat([k_nope, k_rope.expand(-1, -1, self.n_h, -1)], dim=-1)  # [B, T_kv, n_h, qk_head]
        q_full = torch.cat([q_nope, q_rope], dim=-1)                # [B, T, n_h, qk_head]

        # SDPA — FlashAttention-3 in practice. Expects [B, n_h, T, d_head].
        out = F.scaled_dot_product_attention(
            q_full.transpose(1, 2),                                 # [B, n_h, T, qk_head]
            k.transpose(1, 2),                                      # [B, n_h, T_kv, qk_head]
            v.transpose(1, 2),                                      # [B, n_h, T_kv, v_head]
            attn_mask=attn_mask, is_causal=(kv_cache is None),
        ).transpose(1, 2)                                           # [B, T, n_h, v_head]

        return self.wo(out.reshape(B, T, self.n_h * self.v_head))
        # → [B, T, d]


# === Block ================================================================

class DecoderLayer(nn.Module):
    """Role: one transformer block. Two sublayers (attention, FFN), each pre-norm + residual.
    For MoE models, swap `SwiGLU` below for `MoELayer` (see [[MoE Architecture]])."""
    def __init__(self, d, n_h, q_lora, kv_lora, qk_nope, qk_rope, v_head, d_ff):
        super().__init__()
        self.attn_norm = RMSNorm(d)
        self.attn      = MLA(d, n_h, q_lora, kv_lora, qk_nope, qk_rope, v_head)
        self.ffn_norm  = RMSNorm(d)
        self.ffn       = SwiGLU(d, d_ff)

    def forward(self, x, freqs_cis, attn_mask, kv_cache=None):
        # x: [B, T, d]
        h = x + self.attn(self.attn_norm(x), freqs_cis, attn_mask, kv_cache)   # [B, T, d]
        return h + self.ffn(self.ffn_norm(h))
        # → [B, T, d]


# === Input / output ends ==================================================

class Embeddings(nn.Module):
    """Role: token IDs → hidden vectors. Lookup table only — positional info comes later via RoPE inside MLA."""
    def __init__(self, V, d):
        super().__init__()
        self.tok = nn.Embedding(V, d)                              # weight: [V, d]

    def forward(self, ids):
        # ids: [B, T]   (long)
        return self.tok(ids)
        # → [B, T, d]


class LMHead(nn.Module):
    """Role: hidden vectors → next-token logits. Final RMSNorm + linear projection to vocab.
    Weight-tying with the input embedding is common in small models; DeepSeek-V3 and Kimi K2 keep them separate
    (the param savings are negligible at MoE scale and untied gives slightly better quality)."""
    def __init__(self, d, V):
        super().__init__()
        self.norm = RMSNorm(d)
        self.proj = nn.Linear(d, V, bias=False)                    # weight: [V, d]

    def forward(self, x):
        # x: [B, T, d]
        return self.proj(self.norm(x))
        # → [B, T, V]    (logits; softmax done outside, usually fused with the loss)


# === Top-level pipeline ===================================================

class LLM(nn.Module):
    """Role: stitch the pieces. ids [B,T] → embed [B,T,d] → L × DecoderLayer [B,T,d] → LMHead [B,T,V].
    During training all T positions are computed in parallel; during decode T=1 and kv_cache grows by one row per step."""
    def __init__(self, V, d, L, **layer_kw):
        super().__init__()
        self.embed  = Embeddings(V, d)
        self.layers = nn.ModuleList([DecoderLayer(d, **layer_kw) for _ in range(L)])
        self.head   = LMHead(d, V)

    def forward(self, ids, freqs_cis, attn_mask, kv_cache=None):
        # ids: [B, T]
        h = self.embed(ids)                                        # [B, T, d]
        for layer in self.layers:
            h = layer(h, freqs_cis, attn_mask, kv_cache)           # [B, T, d]
        return self.head(h)
        # → [B, T, V]
```

Backward isn't shown — every op above is a standard `nn.Module`, so autograd derives `.backward()` from the fwd graph. The interesting backward shows up in MoE, where token routing introduces explicit collective ops; see [[MoE Architecture]] for that.

→ runtime behavior: [[Inference]], [[Training]].
