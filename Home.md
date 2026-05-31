# Home

Notes for figuring out a paper direction around **RL training for MoE models on heterogeneous GPU clusters**, with SWE-RL as the target workload.

→ **Current plan:** [[Research Plan]] — throughput-optimal async MoE RL on heterogeneous GPU clusters for agentic workloads.

Meta-context (research direction, papers being read, related-work catalog) lives in the agent's memory, not in this vault. The vault is for technical knowledge that accrues as I read.

## Foundations (spine)
- [[Transformer]] — decoder-only block, where params and compute live
- [[Inference]] — prefill, decode, KV cache
- [[Training]] — memory cost, parallelism axes

## MoE (spine)
- [[MoE Architecture]] — sparse FFN, routing, modern design choices
- [[MoE vs Dense Workload]] — compute / memory / comms / hetero interaction — the centerpiece

## Deep Dives — Architecture (lookup material)
- [[Attention Variants|Attention Variants]] — MHA / MQA / GQA / MLA + FlashAttention v1–v3
- [[FFN Variants and Other Sublayers|FFN, Norms, RoPE, ALiBi]] — small sublayer choices, grouped
- [[MoE Routing|MoE Routing]] — top-k, expert choice, shared / fine-grained experts
- [[MoE Load Balancing|MoE Load Balancing]] — aux loss, bias-based, collapse

## Deep Dives — Systems (essential reference)
- [[Dense Training Systems|Dense Training Systems]] — FSDP/ZeRO, TP, PP, recompute, FP8
- [[Dense Inference Systems|Dense Inference Systems]] — PagedAttention, RadixAttention, disaggregation, speculative decoding
- [[MoE Training Systems|MoE Training Systems]] — all-to-all, EP composition, MegaBlocks/Tutel/FlashMoE, hetero MoE
- [[MoE Inference Systems|MoE Inference Systems]] — EP inference, expert offloading, hot-expert replication, RL-on-MoE stacks

## Bibliography
- [[Index|Papers Index]] — downloaded PDFs and scaffold notes by topic

## Papers
*Per-paper scaffold notes live in `Papers/`.*

## Conventions
- Tight notes. No bloat. One idea per page.
- Links = real dependencies, not "see also."
- A dangling `[[link]]` means "this note should exist later." Used sparingly.
