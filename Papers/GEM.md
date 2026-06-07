---
paper_id: "arxiv:2605.19945"
title: "GEM: GPU-Variability-Aware Expert-to-GPU Mapping for Mixture-of-Experts Models"
year: 2026
topic: "moe-serving-systems"
status: "read"
priority: "novelty-boundary"
pdf: "PDFs/moe-serving-systems/2605.19945__gem.pdf"
source_url: "https://arxiv.org/abs/2605.19945"
aliases:
  - "2605.19945"
  - "GEM"
---

# GEM

[PDF](PDFs/moe-serving-systems/2605.19945__gem.pdf) · [Source](https://arxiv.org/abs/2605.19945)

> **THE most dangerous neighbor for [[Research Plan 2]] (TierShift) — read the carve-out carefully.**
> GEM is GPU-variability-aware MoE expert→GPU mapping with the *same straggler motivation*. It does
> **not** kill TierShift — but it forces the novelty claim to be precise. Cite it front-and-center.

## TL;DR
At the MoE all-to-all barrier, the **straggler GPU** (the one that finishes its experts last) caps
per-layer latency. Prior placement balances *token load* but ignores **GPU variability** and can land
hot experts on the slowest GPU. GEM profiles each GPU's speed + per-task token-load distribution and
computes an **expert→GPU mapping** that gives faster GPUs *proportionally more* tokens so all finish
together. 7.9% avg (≤16.5%) end-to-end latency; 9.1% avg p90 TPOT. Tested on Mixtral-8x7B/8x22B,
Llama-4-Scout, Hunyuan-A13B, **Qwen3-30B-A3B**.

## The carve-out (why TierShift survives — four axes, all load-bearing)
1. **Static vs. online.** GEM *"loads each expert's weights onto its assigned GPU at model load time
   and continues to use it throughout the deployment"* — mapping computed **once** from an early
   trace, never adapted. **It names TierShift's wedge as its own limitation:** *"the static mapping
   does not account for distribution shift in token routing patterns post-deployment."* TierShift
   adapts to drift online.
2. **Intra-generation variability vs. cross-architecture tier gap.** GEM's variability is **7.7–27.7%
   among *identical* L40s** (process variation, DVFS) — explicitly *"within modern GPUs, not
   comparing fundamentally different hardware tiers."* TierShift's gap is the **2.4–4× architectural**
   fast/slow (Ada vs Ampere, FP8-vs-none). Different magnitude *and* different physics.
3. **No replication / no token-split.** GEM keeps **each expert on exactly one GPU** ("number of
   experts per GPU = total / #GPUs"). TierShift's **replicate-and-split** (Layer 2) is precisely what
   GEM does *not* do — GEM achieves speed-proportional load at *placement* granularity (coarse,
   static, one-expert-one-GPU); TierShift at *token-split* granularity (fine, online, one-expert-
   across-tiers).
4. **Latency-only vs. p99-SLO + $/token.** GEM minimizes straggler latency + p90 TPOT. TierShift adds
   the **cost objective on a mixed fleet** (hold p99 at min $/token).

→ Sharpened claim (use this): *existing work does **static/profiled** heterogeneity-aware mapping
(GEM, Aurora), **homogeneous** online replication/prefetch ([[CRAFT]], [[HarMoEny]], PROBE), or
**memory-tier** expert caching ([[Toward-Efficient-MoE-Inference]]). TierShift is **online tier-aware
placement + migration + replicate-and-split** under p99/$/token, with an explicit migration-timescale
**regime map.*** Do **not** claim "nobody has done hetero expert mapping."

## CRITICAL planning signal (changes B2's model choice)
GEM reports **Qwen3-30B-A3B benefits *least* (~1.5%)** because of *"near-uniform routing across 128
small experts."* **Fine-grained MoE has flat routing → little skew → little to exploit.** Skew (hence
TierShift's whole premise) lives in **coarser-routing models** (Mixtral-8x7B, OLMoE-64e). → OLMoE is a
*good* skew testbed; **Qwen3-30B-A3B may show weak drift** — reconsider it as the scale-up target, or
add a Mixtral-class model. This is a direct input to [[Initial Plan 2]] Risk-2.

## GEM's other useful gifts
- **"Consistent" vs "temporal" experts:** consistent = used ~85% of timesteps (caught by average);
  temporal = used ~17% of timesteps but 3× tokens when active, **correlated/bursty**. Validates that
  hot-set non-stationarity is real — and that *temporal* experts are exactly what an online policy
  (not GEM's static map) should chase.
- Profiling cost: variability profile 2.13 min, mapping 8.8 s (Llama-4-Scout). 16-timestep trace window.
- Eval limited to **4 GPUs with emulated variability**; large-scale untested (their stated gap).

## Links
- [[HarMoEny]] (dynamic token redistribution, homogeneous) · [[Aurora]] (static hetero, theory) · [[MegaScale-Infer]] · [[CRAFT]]-style replication · [[Toward-Efficient-MoE-Inference]] · [[Research Plan 2]]
