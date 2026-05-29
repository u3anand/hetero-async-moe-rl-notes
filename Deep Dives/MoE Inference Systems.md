# MoE Inference Systems

Systems perspective on serving and rolling out MoE models. Builds on [[Dense Inference Systems]] + the all-to-all picture from [[MoE Training Systems]].

## Per-token routing breaks dense batching assumptions

For a dense model, a batch of `B` tokens runs the same compute on every token. For MoE, different tokens hit different experts → expert utilization within a batch is non-uniform.

Consequences:
- **Grouped GEMM** — group tokens by expert assignment, run one GEMM per expert. Standard kernel pattern.
- **Imbalanced expert load per batch** — if 80% of tokens go to 3 experts, those experts' GEMMs bottleneck the whole batch. Mitigation: larger batch size so imbalance averages out (LLN).
- **Token padding to capacity** — fixed-tensor-shape kernels pad each expert to max load → wasted compute. FlashMoE / DeepEP eliminate this with variable-shape kernels.

## Expert parallelism (EP) for inference

Same EP idea as training (see [[MoE Training Systems]]) but bwd-free. Each rank holds a subset; all-to-all dispatch + combine per MoE layer.

- **vLLM** — production EP support; EP within a node (NVLink), DP across nodes.
- **SGLang** — DeepSeek-V3 production stack uses SGLang fused MoE kernels + EP.
- **TensorRT-LLM** — NVIDIA's stack; custom kernels for MoE; used for DeepSeek-V3 serving at scale.

For long-context inference, **EP + RadixAttention prefix caching** is the workhorse combination.

## Expert offloading (single-GPU / memory-constrained)

When the full expert set doesn't fit in HBM:

- **Fiddler (ICLR 2025)** — keep all experts in CPU memory; when a token needs an expert, transfer the **activation** (not the weights) to CPU, compute there, return. ~10× latency reduction vs prior offloading by avoiding weight transfer.
- **eMoE / Mixtral offload patterns** — predict which experts are likely needed, prefetch weights.
- **Layer-wise expert swapping** — hold one MoE layer's experts in HBM, swap as fwd progresses.

Mostly useful for memory-poor inference (consumer GPUs); not the production-cluster regime.

## Hot-expert replication

When a few experts dominate routing (see [[MoE Load Balancing]]), the EP step is bound by the hot-expert rank. Solutions:

- **Replicate hot experts** across multiple ranks; route a fraction of their traffic to each replica. Cuts per-rank load.
- **Static replication** — measure routing stats over a window, replicate the top-N experts.
- **Dynamic replication** — re-place experts as the routing distribution drifts (rare, expensive).

Replication usually beats migration because routing drifts on minutes-to-hours scale, not seconds. **Open issue under async RL:** replica synchronization when different rollout replicas update at different rates.

## Production stacks for MoE RL serving

- **slime** (THUDM) — RL framework for MoE; powers GLM-4.5/4.6/4.7 RL. Built on Megatron training + SGLang inference. Homogeneous clusters only.
- **prime-rl** (Prime Intellect) — open MoE RL framework. FSDP2 + vLLM + EP + prefill/decode disaggregation. Has a Qwen3-30B-A3B-SWE recipe.
- **Salesforce SFR-RL** — 1T+ MoE RL on 1000+ GPUs. "Least-loaded expert parallelism" — schedule traffic to under-loaded expert replicas.
- *Each Prompt Matters* (Prime Intellect, 2025) — efficient rollout scheduling for RL on 100B+ MoE; doesn't waste rollouts on hard prompts.
- *Stabilizing MoE RL by Aligning Training and Inference Routers* (2025) — RL-specific routing-drift fix.

## Why this matters for the research direction

(See agent memory: `research-direction`.)

- All three production stacks assume homogeneous clusters. Their EP collapses on hetero links.
- Hot-expert replication composes awkwardly with prefix-caching (different prefixes → different routing distributions → different replication targets).
- Async RL adds asymmetric routing-distribution staleness across replicas. R3-style replay only partially addresses this.

## Sources

- *Fiddler: CPU-GPU Orchestration for Fast Inference of MoE Models* (ICLR 2025)
- *Each Prompt Matters* (Prime Intellect, 2025) — [arXiv:2512.07710](https://arxiv.org/abs/2512.07710)
- *Stabilizing MoE RL by Aligning Training and Inference Routers* (2025) — [arXiv:2510.11370](https://arxiv.org/abs/2510.11370)
- Repo: *slime* (THUDM) — [github.com/THUDM/slime](https://github.com/THUDM/slime)
- Blog: Salesforce, *Building Efficient RL Training for the Agentic Era* — [salesforce.com/blog/efficient-rl-training-agentic-era](https://www.salesforce.com/blog/efficient-rl-training-agentic-era/)
- Doc: vLLM Expert Parallel Deployment — [docs.vllm.ai/en/latest/serving/expert_parallel_deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- Blog: *The vLLM MoE Playbook* (AMD ROCm) — [rocm.blogs.amd.com/.../vllm-moe-guide/README.html](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
