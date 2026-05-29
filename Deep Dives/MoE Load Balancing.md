# MoE Load Balancing

Why MoE needs explicit balancing and the variants used to achieve it.

## The problem

Token-choice top-k routing (see [[MoE Routing]]) creates a feedback loop: if expert A gets more tokens, it specializes faster, gets even more tokens. A few experts dominate; others starve; the model effectively shrinks.

Unbalanced load also kills throughput: in [[MoE Training Systems|expert parallelism]], the rank holding the busiest expert stalls everyone.

## Aux-loss balancing (classic)

Add a term to the training loss that penalizes uneven utilization. Most common form:

```
L_aux = α · N · sum_i (f_i · P_i)
```

- `f_i` = fraction of tokens routed to expert `i` in the batch
- `P_i` = mean routing probability assigned to expert `i`
- `α` = balancing coefficient (typically 0.01)

Minimized when both `f` and `P` are uniform. Used in Switch, GShard, Mixtral.

Downside: perturbs the task loss → trades task quality for balance. The `α` is finicky.

## Bias-based balancing (DeepSeek-V3, "aux-loss-free")

DeepSeek-V3 ditches the aux loss. Instead, maintain a per-expert **bias** added to router logits **for routing only** (not used to weight outputs). Update online: if expert `i` overloaded, decrement its bias; if underloaded, increment.

```
routing_logit_i = router(x)_i + bias_i        # used for top-k selection
expert_weight_i = softmax(router(x))_i        # used for output weighting; gradient flows here
```

The bias slowly steers tokens away from overloaded experts without contaminating the loss. Cleaner; doesn't trade off quality for balance. DeepSeek-V3 demonstrates this at 671B-A37B with near-perfect balance and no aux loss.

## Router-z loss

Separate from balancing — stabilizes router training by penalizing router logit magnitudes:

```
L_z = β · (logsumexp(router_logits))²
```

Keeps decisions confident but not blown up. Used in ST-MoE, OLMoE.

## Capacity factor (the dropping safety net)

In distributed training, each expert is allocated a fixed buffer per batch. Tokens beyond an expert's capacity are **dropped** (skip the FFN, residual passes through). Capacity factor (typically 1.0–1.5) controls overflow tolerance vs throughput.

Inference usually disables dropping (capacity = ∞).

## Specialization vs balance — the genuine tradeoff

Perfect balance (every expert sees equal load) defeats the point of MoE — experts should specialize. *Some* imbalance is good; it's signal of real specialization. The art is preventing **pathological** imbalance (collapse, dead experts) without forcing uniformity.

Bias-based balancing handles this well — it nudges away from collapse without flattening the distribution to uniform.

## Why this matters for the research direction

(See agent memory: `research-direction`.)

- Async RL across rollout replicas: bias values are model state — they must be synced or replicas diverge in routing. Open issue.
- Hot-expert hotspot is a hardware problem (slowest EP rank wins each step). Balancing reduces it but doesn't eliminate it — hence **hot-expert replication** as a systems-side mitigation (see [[MoE Inference Systems]] and [[MoE Training Systems]]).

## Sources

- *Switch Transformer* (Fedus et al., 2021) — [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
- *DeepSeek-V3* (2024, bias-based balancing) — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- *OLMoE* (Muennighoff et al., 2024, router-z loss) — [arXiv:2409.02060](https://arxiv.org/abs/2409.02060)
- Blog: Cerebras, *Router Wars: Which MoE Routing Strategy Actually Works* — [cerebras.ai/blog/moe-guide-router](https://www.cerebras.ai/blog/moe-guide-router)
