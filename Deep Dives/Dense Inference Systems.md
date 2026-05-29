# Dense Inference Systems

Systems perspective on serving dense LLMs. KV cache management, batching, disaggregation, decoding tricks.

## KV cache management

The KV cache (see [[Inference]]) is per-request and grows with context. Serving needs to manage it like a memory allocator.

- **PagedAttention (vLLM)** — split KV cache into fixed-size blocks (typically 16 tokens). Each request holds a list of block pointers; blocks are non-contiguous in HBM. Solves fragmentation; enables block-level sharing across requests. Eliminates 60–80% of memory waste.
- **RadixAttention (SGLang)** — extends PagedAttention with a radix tree over prompt prefixes. Common prefixes (system prompts, in-context examples, SWE repo state) share cached blocks across requests. 2–5× throughput on prefix-heavy workloads.
- **Block sharing for tree search / beam / parallel sampling** — multiple sequences from the same prefix share blocks.

For SWE-RL specifically, RadixAttention's prefix sharing is the natural fit: many rollouts share the same repo prefix.

## Continuous batching (iteration-level scheduling)

Static batching: pad all requests to longest, run together until all finish. Wastes compute when sequences finish at different times.

**Continuous batching** (Orca, vLLM): at each decoding step, the scheduler picks which requests to step. Finished requests leave; new ones join. Each iteration's batch composition is dynamic. 5–10× throughput on mixed-length workloads.

## Chunked prefill

Prefill is compute-bound; decode is memory-bound (see [[Inference]]). Co-locating them naively: a long prefill stalls decode for all concurrent requests.

**Chunked prefill**: split a long prefill into N-token chunks, interleave with decode steps. Keeps decode latency low while still making prefill progress. vLLM and SGLang both support this. Tunable token budget per scheduler step controls the prefill/decode tradeoff.

## Prefill/decode disaggregation

Run prefill on one set of GPUs ("prefill workers"), decode on another ("decode workers"). KV cache is transferred from prefill to decode worker after prefill.

- **Splitwise** (MSR, 2023) — first formal proposal.
- **DistServe** (OSDI 2024) — formalized goodput-optimized scheduling; ~7× over co-located.
- **Mooncake** (Moonshot AI / Kimi, 2024) — production KV-cache-centric architecture; separated cache layer (CPU/SSD); cross-DC capable.

Why: prefill GPUs and decode GPUs have different sweet spots (compute-bound vs bandwidth-bound). Disaggregation lets each scale independently. Composes naturally with async RL rollouts: a fast tier could handle prefill, a slow tier could handle decode (relevant for the open seams in agent memory `research-direction`).

## Speculative decoding

A small **draft model** generates `k` tokens cheaply; the target model verifies them in one forward pass; accepted tokens commit. ~2–4× decode speedup at no quality cost.

Variants:
- **Vanilla speculative decoding** (Leviathan, Chen, 2023) — separate draft model.
- **Medusa** — extra decoding heads on the target model itself; no separate draft.
- **EAGLE / EAGLE-2 / EAGLE-3** — feature-level draft; SOTA at 3–6× speedup.

Tradeoff: more aggressive drafting = higher rejection cost. Acceptance rate is workload-dependent.

## Quantization (briefly)

- **W8A8** — INT8 weights + activations. Standard for serving; usually <1% quality drop.
- **W4A16** — INT4 weights, BF16 activations (AWQ, GPTQ). Consumer deployment.
- **W4A4 / FP8 inference** — emerging on H100/B200. Requires careful calibration.

## Implications for RL rollouts

For SWE-RL on hetero clusters:
- RadixAttention prefix sharing → big win across rollouts of the same repo.
- Chunked prefill → keeps decode latency low during long repo prefills.
- Disaggregation → fast-tier prefill, slow-tier decode is a candidate structure.

The MoE-specific extensions (per-token routing, expert co-location) are in [[MoE Inference Systems]].

## Sources

- *PagedAttention / vLLM* (Kwon et al., OSDI 2023) — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- *SGLang / RadixAttention* (Lian et al., 2023) — [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- *DistServe* (Zhong et al., OSDI 2024) — [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
- *Mooncake* (Qin et al., 2024) — [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
- *EAGLE* (Li et al., ICLR 2024) — [openreview.net](https://openreview.net/pdf?id=1NdN7eXyb4)
- Blog: vLLM team, *Inside vLLM: Anatomy of a High-Throughput LLM Inference System* — [blog.vllm.ai/2025/09/05/anatomy-of-vllm.html](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- Blog: LMSYS, *Fast and Expressive LLM Inference with RadixAttention and SGLang* — [lmsys.org/blog/2024-01-17-sglang](https://www.lmsys.org/blog/2024-01-17-sglang/)
