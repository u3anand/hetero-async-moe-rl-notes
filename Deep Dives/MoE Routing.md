# MoE Routing

Variants of the per-token routing decision in MoE layers, plus the structural choices around expert composition.

## Routing schemes

**Top-k routing** (default). For each token, the router (a linear layer) produces logits over experts; softmax → pick top-k experts; weight expert outputs by router probabilities.

- **k = 1** (Switch Transformer) — simplest, max throughput, harder to train.
- **k = 2** (GShard, Mixtral) — standard.
- **k = 4–8** (DeepSeek-V2/V3 fine-grained) — more experts per token, each smaller.

Top-k is **token-choice**: each token picks its experts independently. This creates the load-balancing problem (see [[MoE Load Balancing]]).

**Expert choice routing** (Google, 2022) — flip it: each expert picks the top-N tokens it wants. Eliminates routing imbalance by construction (every expert gets exactly N tokens). Cost: each token gets a variable number of experts (some get zero — dropped tokens). Less common in current open MoEs because token dropping is awkward for autoregressive generation. Still a clean idea for pretraining.

**Hash routing** — fixed, content-independent assignment of tokens to experts (e.g. hash of token ID). Trivial to balance; no learned specialization. Used as a baseline in Switch.

## Shared experts (DeepSeek design)

DeepSeek-V2/V3 add **always-on shared experts** in parallel to routed experts:

```
output = sum_i router_weight_i · routed_expert_i(x)  +  shared_expert(x)
```

Rationale: common features (syntax, basic patterns) shouldn't compete with specialized features for expert slots. Letting shared experts absorb the common case frees routed experts to specialize.

Empirically: shared experts + fine-grained routed experts is the SOTA combination as of 2025 (DeepSeek-V3, Qwen3-MoE).

## Fine-grained experts

Move from "few big experts" to "many small experts."

- Mixtral 8×7B → 8 experts, each large.
- DeepSeek-V3 → 256 routed experts + 1 shared, each small.
- Qwen3-235B-A22B → 128 experts.
- OLMoE → 64 experts.

Same active params per token, but more specialization room. Cost: more routing decisions, more all-to-all communication granularity (see [[MoE Training Systems]]).

## Routing collapse & train/inference drift

**Routing collapse**: all tokens route to the same few experts → others starve, then get effectively zero gradient → collapse worsens. Aux losses ([[MoE Load Balancing]]) try to fix this during training.

**Train/inference routing divergence**: during RL training (or distillation), the routing distribution at inference time drifts from training time. Tokens go to different experts → KL between train and inference policies grows. The "Rollout Routing Replay" (R3) paper addresses this by recording inference routing distributions and replaying them during training.

This is one of the live open issues for RL on MoE (see agent memory: `research-direction`).

## Why this matters for the research direction

- Fine-grained experts → more all-to-all volume per layer → worse on hetero clusters.
- Shared experts → some FFN compute is always-on (more like dense) → easier to schedule on a fast tier.
- Train/inference routing divergence is **acute under async RL** because rollout replicas update at different rates → each replica has its own drift trajectory.

## Sources

- *Switch Transformer* (Fedus et al., 2021) — [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
- *GShard* (Lepikhin et al., 2020) — [arXiv:2006.16668](https://arxiv.org/abs/2006.16668)
- *Expert Choice Routing* (Zhou et al., NeurIPS 2022) — [arXiv:2202.09368](https://arxiv.org/abs/2202.09368)
- *DeepSeek-V3* (2024) — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- *OLMoE* (Muennighoff et al., 2024) — [arXiv:2409.02060](https://arxiv.org/abs/2409.02060)
- Blog: Cameron R. Wolfe, *Conditional Computation: The Birth of MoE* — [cameronrwolfe.substack.com](https://cameronrwolfe.substack.com/p/conditional-computation-the-birth)
- Blog: HuggingFace, *Mixture of Experts Explained* — [huggingface.co/blog/moe](https://huggingface.co/blog/moe)
