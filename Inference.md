# Inference

Two phases per request.

## Prefill
- Process the full prompt in one forward pass; all token positions in parallel.
- **Compute-bound** (large matmul, high arithmetic intensity).
- Populates KV cache for every layer.
- FLOPs ≈ `2 · N · P_active` (N = prompt tokens, P_active = active params per token).

## Decode
- Generate one token at a time, autoregressive.
- Each step: 1 new token × **full KV-cache read** × full active-weights read.
- **Memory-bandwidth-bound** (arithmetic intensity ≈ 1).
- Step latency ≈ (active weights + KV cache) / HBM bandwidth.

## KV Cache
- Per layer, per token: K and V tensors. Cached to skip recompute on the previous tokens.
- Size per request ≈ `2 · L · N · n_kv_heads · d_head · bytes`.
- **Linear in context length** → dominates memory for long-context serving.
- GQA/MQA reduce `n_kv_heads` (shared K/V across query heads) → cheaper cache.

→ Deep Dives: [[Attention Variants|Attention Variants]] (GQA/MLA shrink the cache further), [[Dense Inference Systems|Dense Inference Systems]] (PagedAttention, RadixAttention, disaggregation, speculative decoding)

## Why this matters for the RL direction
(See agent memory: `research-direction`.)

- **Repo prefill** is the upfront tax in SWE-RL rollouts (large prompt).
- **Decode** is the dominant cost of long-horizon traces (many tool-call turns).
- **Prefix overlap** across rollouts of the same repo → KV-cache reuse opportunity across replicas / tiers.

→ MoE adds another wrinkle: routing varies per-token, so cache reuse interacts with expert co-location (see [[MoE vs Dense Workload]], [[MoE Inference Systems|MoE Inference Systems]]).
