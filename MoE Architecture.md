 # MoE Architecture

Replace each block's dense FFN with a **sparse MoE FFN**. Attention is unchanged.

## Layer

```
x → router (Linear → top-k softmax)
  → top-k experts (each = small FFN)
  → weighted combine
```

## Sizing
- `N` experts, each a small FFN.
- Per-token compute: only `k` experts fire → `P_active ≪ P_total`.
- Example: Qwen3 80B-A3B has 80B total params, ~3B active per token.

## Modern design choices
- **Shared expert** — always-on FFN added in parallel to routed experts (DeepSeek-V2/V3, Qwen3-MoE). Captures common features so routed experts can specialize.
- **Fine-grained experts** — many small experts (64–256) over few big ones. Better specialization, harder routing.
- **Load balancing** — aux loss to flatten expert utilization. DeepSeek-V3 drops aux loss for a bias-based balancing scheme.

→ Deep Dives: [[MoE Routing|MoE Routing]] (top-k, expert choice, shared / fine-grained experts), [[MoE Load Balancing|MoE Load Balancing]] (aux loss vs bias-based, collapse)

## Why this exists
- Decouples model **capacity** (`P_total`) from per-token **compute** (`P_active`).
- An 80B MoE has the per-token FLOPs of a ~3B dense model but the knowledge of something much larger.

## Reference code (Kimi K2 / DeepSeek-V3 structure)

Annotated reference. Structure follows the Kimi K2 and DeepSeek-V3 open releases — sigmoid router with per-expert bias, hierarchical group routing, fine-grained routed experts + one shared expert. Production code adds optimized grouped GEMM kernels (`DeepEP`, `FlashMoE`), expert-tile bucketing, fused dispatch/combine, etc. — stripped here. Upstream:

- Kimi K2: [github.com/MoonshotAI/Kimi-K2](https://github.com/MoonshotAI/Kimi-K2)
- DeepSeek-V3: [github.com/deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
- All-to-all kernel reference: [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)

`SwiGLU` and `RMSNorm` are reused from [[Transformer#Reference code (Kimi K2 / DeepSeek-V3 structure)|Transformer]]. Shape symbols `B`, `T`, `d` mean the same here; new ones below.

**MoE-specific shape symbols**

| symbol | meaning | Kimi K2 | DeepSeek-V3 |
|---|---|---|---|
| `N` | flattened tokens on this rank, `B·T` | — | — |
| `k` | top-k experts per token | 8 | 8 |
| `E` | total routed experts | 384 | 256 |
| `g` | router groups (hierarchical routing) | 1 (no grouping) | 8 |
| `topk_g` | top groups per token | — | 4 |
| `d_e` | per-routed-expert FFN intermediate | ~2048 | 2048 |
| `n_shared` | width multiplier of the always-on shared expert | 1 | 1 |
| `ep_size` | EP world size (how many ranks share `E`) | varies | varies |
| `E_loc` | local experts per rank, `E / ep_size` | — | — |

The shared expert's role is to absorb the common case so the routed experts can specialize — see [[MoE Routing#Shared experts (DeepSeek design)|shared experts]].

### Forward pass — single rank (no EP)

```python
class MoELayer(nn.Module):
    """Role: drop-in replacement for SwiGLU inside a DecoderLayer. Token-conditional FFN compute.

    Two compute paths added in parallel:
      - Routed path: each token picks k experts out of E. Weighted sum of expert outputs.
      - Shared path: an always-on FFN every token goes through. Adds, doesn't replace.

    Bias-based balancing (DeepSeek-V3 / Kimi K2):
      - `routed_bias`  is *added to router logits for top-k selection only*, NOT used to weight outputs.
      - Updated online by `update_bias()` (no grad). Steers traffic away from overloaded experts.
    """
    def __init__(self, d, d_e, E, n_shared, k, g=1, topk_g=None):
        super().__init__()
        self.k, self.E, self.g, self.topk_g = k, E, g, topk_g

        self.router = nn.Linear(d, E, bias=False)                          # weight: [E, d]
        self.register_buffer("routed_bias", torch.zeros(E))                # [E], gradient-free
        self.experts = nn.ModuleList([SwiGLU(d, d_e) for _ in range(E)])   # E experts, each [d → d_e → d]
        self.shared  = SwiGLU(d, d_e * n_shared)                           # one always-on expert, wider

    def forward(self, x):
        # x: [B, T, d]
        B, T, d = x.shape
        x_flat = x.view(B * T, d)                                          # [N, d]   where N = B·T

        # 1. Router scores. Sigmoid (Kimi K2 + DeepSeek-V3); earlier V2 used softmax.
        scores = torch.sigmoid(self.router(x_flat))                        # [N, E]
        biased = scores + self.routed_bias                                 # [N, E]   bias is for selection

        # 2. Hierarchical top-k (only if g > 1). Otherwise: plain top-k over E.
        if self.g > 1:
            # Each group's score = sum of its top-2 experts' biased scores.
            per_g = biased.view(-1, self.g, self.E // self.g)              # [N, g, E/g]
            g_score = per_g.topk(2, dim=-1).values.sum(-1)                 # [N, g]
            top_g   = g_score.topk(self.topk_g, dim=-1).indices            # [N, topk_g]
            g_mask  = torch.zeros_like(g_score).scatter_(1, top_g, 1.0)    # [N, g]   1 on selected groups
            # Expand group mask to expert mask; -inf out non-selected experts.
            e_mask  = g_mask.repeat_interleave(self.E // self.g, dim=-1)   # [N, E]
            biased  = biased.masked_fill(e_mask == 0, float("-inf"))

        topk_idx = biased.topk(self.k, dim=-1).indices                     # [N, k]   chosen expert IDs
        weights  = scores.gather(1, topk_idx)                              # [N, k]   from *unbiased* scores
        weights  = weights / weights.sum(-1, keepdim=True)                 # [N, k]   renormalize

        # 3. Dispatch + run + combine. Single-rank loop (replaced by EPDispatchCombine for EP).
        out = torch.zeros_like(x_flat)                                     # [N, d]
        for e in range(self.E):
            sel = (topk_idx == e)                                          # [N, k]   bool
            if not sel.any():
                continue
            tok, choice = sel.nonzero(as_tuple=True)                       # tok, choice: [n_e]
            expert_out  = self.experts[e](x_flat[tok])                     # [n_e, d]
            out.index_add_(0, tok, weights[tok, choice].unsqueeze(-1) * expert_out)

        # 4. Shared expert (always-on, no routing).
        out = out + self.shared(x_flat)                                    # [N, d]
        return out.view(B, T, d), topk_idx
        # → [B, T, d], [N, k]   (topk_idx returned so update_bias can be called by the training loop)

    @torch.no_grad()
    def update_bias(self, topk_idx, gamma=1e-3):
        """Role: keep expert load roughly balanced. Runs after each step, gradient-free.
        Nudges overloaded experts' biases down; underloaded up. The bias only affects *selection*."""
        # topk_idx: [N, k]
        load = torch.bincount(topk_idx.flatten(), minlength=self.E).float()  # [E]
        err  = load - load.mean()                                            # [E]
        self.routed_bias.sub_(gamma * err.sign())                            # [E]
```

### Forward + backward across EP ranks

Once experts are sharded across GPUs (Expert Parallelism — see [[MoE Training Systems|MoE Training Systems]]), step 3 of `MoELayer.forward` becomes two all-to-alls per direction. Standard PyTorch autograd doesn't see across `dist.all_to_all`, so EP is wrapped in a `torch.autograd.Function` that defines its own backward.

```python
class EPDispatchCombine(torch.autograd.Function):
    """Role: one MoE layer's communication. Encapsulates dispatch → local experts → combine.

    Forward (2 all-to-alls):
        permute tokens by destination rank → all-to-all DISPATCH → tokens arrive at expert-holding ranks
        run local experts on received tokens
        all-to-all COMBINE → results return to home ranks → un-permute → weighted sum across k picks

    Backward (2 mirrored all-to-alls — counts swap, direction reverses):
        un-permute grad_out, broadcast across k → all-to-all (combine⁻¹: send grads to expert holders)
        autograd backprops through local experts (their weights get .grad in the usual way)
        all-to-all (dispatch⁻¹: send input-grads back to home ranks) → un-permute → sum across the k picks

    Total: 4 all-to-alls per MoE layer per training step.
    """

    @staticmethod
    def forward(ctx, x, topk_idx, weights, local_experts, ep_group, E):
        # x:        [N, d]           — local tokens on this rank
        # topk_idx: [N, k]            — global expert IDs each token picked
        # weights:  [N, k]            — already renormalized
        # local_experts: list of E_loc SwiGLUs held on this rank
        ep_size = dist.get_world_size(ep_group)
        rank    = dist.get_rank(ep_group)
        E_loc   = E // ep_size
        N, k    = topk_idx.shape
        d       = x.size(-1)

        # --- 0. Lay out sends. Each token replicated k times → one row per (token, choice).
        flat_idx  = topk_idx.flatten()                                     # [N·k]
        dest_rank = flat_idx // E_loc                                      # [N·k]   which rank holds this expert
        local_eid = flat_idx %  E_loc                                      # [N·k]   which slot on that rank
        perm      = dest_rank.argsort()                                    # [N·k]   group sends by rank
        send_buf  = x.repeat_interleave(k, dim=0)[perm]                    # [N·k, d]
        send_eid  = local_eid[perm]                                        # [N·k]
        send_cnt  = torch.bincount(dest_rank, minlength=ep_size)           # [ep_size]   per-dest counts

        # --- 1. All-to-all DISPATCH: tokens travel to expert holders.
        recv_cnt = torch.empty_like(send_cnt)                              # [ep_size]
        dist.all_to_all_single(recv_cnt, send_cnt, group=ep_group)         # exchange counts first
        R = int(recv_cnt.sum())                                            # received row count (varies per step!)
        recv_buf = torch.empty((R, d), dtype=x.dtype, device=x.device)     # [R, d]
        dist.all_to_all_single(recv_buf, send_buf,
                               output_split_sizes=recv_cnt.tolist(),
                               input_split_sizes=send_cnt.tolist(), group=ep_group)
        recv_eid = torch.empty_like(send_eid[:R])                          # [R]
        dist.all_to_all_single(recv_eid, send_eid,
                               output_split_sizes=recv_cnt.tolist(),
                               input_split_sizes=send_cnt.tolist(), group=ep_group)

        # --- 2. Local experts. Production: one grouped GEMM per local expert. Loop here for clarity.
        out_buf = torch.empty_like(recv_buf)                               # [R, d]
        for e, expert in enumerate(local_experts):
            sel = (recv_eid == e)                                          # [R]
            if sel.any():
                out_buf[sel] = expert(recv_buf[sel])                       # [n_e, d]

        # --- 3. All-to-all COMBINE: results return home (counts swap vs dispatch).
        combined = torch.empty_like(send_buf)                              # [N·k, d]
        dist.all_to_all_single(combined, out_buf,
                               output_split_sizes=send_cnt.tolist(),
                               input_split_sizes=recv_cnt.tolist(), group=ep_group)

        # --- 4. Un-permute → [N, k, d]; weight + sum across the k picks.
        inv       = perm.argsort()                                         # [N·k]
        per_token = combined[inv].view(N, k, d)                            # [N, k, d]
        out       = (per_token * weights.unsqueeze(-1)).sum(dim=1)         # [N, d]

        ctx.save_for_backward(perm, weights, recv_eid)
        ctx.send_cnt, ctx.recv_cnt = send_cnt, recv_cnt
        ctx.ep_group, ctx.local_experts = ep_group, local_experts
        ctx.shape = (N, k, d)
        return out                                                          # [N, d]

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out: [N, d]
        perm, weights, recv_eid = ctx.saved_tensors
        ep_group = ctx.ep_group
        N, k, d  = ctx.shape

        # --- 1. dL/d(per_token_choice) = grad_out · weight ; broadcast to (N, k, d) → send-order.
        grad_choices  = grad_out.unsqueeze(1) * weights.unsqueeze(-1)      # [N, k, d]
        grad_combined = grad_choices.reshape(-1, d)[perm]                   # [N·k, d]   send-order

        # --- 2. Mirror of COMBINE: grads flow back along the combine path (counts swap vs fwd-combine).
        R = int(ctx.recv_cnt.sum())
        grad_recv = torch.empty((R, d), dtype=grad_out.dtype, device=grad_out.device)  # [R, d]
        dist.all_to_all_single(grad_recv, grad_combined,
                               output_split_sizes=ctx.recv_cnt.tolist(),
                               input_split_sizes=ctx.send_cnt.tolist(), group=ep_group)

        # --- 3. Backprop through each local expert.
        # Each expert's forward built a small autograd graph; we differentiate it w.r.t. its inputs,
        # and weight-grads accumulate into expert.parameters().grad as a side effect.
        grad_local = torch.empty_like(grad_recv)                            # [R, d]
        for e, expert in enumerate(ctx.local_experts):
            sel = (recv_eid == e)
            if sel.any():
                grad_local[sel] = torch.autograd.grad(
                    outputs=expert.last_output,                             # [n_e, d]
                    inputs =expert.last_input,                              # [n_e, d]
                    grad_outputs=grad_recv[sel],                            # [n_e, d]
                    retain_graph=False,
                )[0]

        # --- 4. Mirror of DISPATCH: input-grads return to home ranks.
        grad_send = torch.empty_like(grad_combined)                         # [N·k, d]
        dist.all_to_all_single(grad_send, grad_local,
                               output_split_sizes=ctx.send_cnt.tolist(),
                               input_split_sizes=ctx.recv_cnt.tolist(), group=ep_group)

        # --- 5. Un-permute → [N, k, d]; sum the k contributions per home token.
        inv    = perm.argsort()
        grad_x = grad_send[inv].view(N, k, d).sum(dim=1)                    # [N, d]
        return grad_x, None, None, None, None, None
        # Only `x` has a grad. topk_idx (long), weights (constant in this sketch),
        # local_experts, ep_group, E carry no grads.
```

Two things this makes concrete:

- **4 all-to-alls per MoE layer per training step** — dispatch + combine in fwd, combine⁻¹ + dispatch⁻¹ in bwd. The collective load that pins EP groups across ranks. Optimized kernels (DeepEP, FlashMoE) shrink bandwidth waste, not the count. See [[MoE Training Systems#All-to-all volume and topology|all-to-all volume]].
- **`routed_bias` is model state, not a gradient quantity** — `update_bias` runs in `no_grad`. Under async RL, this state has to be sync'd across rollout replicas the same way weights are, or each replica drifts its own routing distribution. Open issue (see [[MoE Routing#Routing collapse & train/inference drift|routing drift]]).

## What it breaks
- **Routing is token-conditional** → expert load varies per-batch, drifts over training.
- **Memory** ≈ `P_total` (all experts must be reachable for the next token's routing).
- Adds **all-to-all** comms in fwd and bwd → see [[MoE vs Dense Workload]].
- Train/inference routing divergence is a known problem (cf. Rollout Routing Replay).

## What gets sharded across GPUs
- **Expert Parallelism (EP)**: experts partitioned across ranks. Each rank holds a subset; tokens are dispatched to the rank holding their chosen expert.
- Hot/cold imbalance becomes a hardware problem, not just a learning problem — see [[MoE vs Dense Workload]].
