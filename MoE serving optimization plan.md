# MoE serving optimization plan

An online **multi-tier expert cache** for MoE inference on heterogeneous GPUs.

## Motivation
- MoE dominates open LLMs (Mixtral, Qwen3, DeepSeek); served via **expert parallelism (EP)**.
- Expert load is **skewed and non-stationary** — the hot set drifts with the request mix.
  (B2/OLMoE: imbalance ratio 2.3→8.0, shifts by domain, reactive hit-rate 0.83.)
- On a **fast+slow GPU** fleet, a hot expert on the slow tier is a **straggler** at the all-to-all
  barrier. Correct placement is hotness- *and* tier-aware — and it moves.
- **Goal:** lower **$/token** on a mixed fleet vs. all-fast. *Online* target = hold a p99 SLO;
  *offline/batch* target = max throughput/$ 

## Main idea
- Treat experts as a **3-tier cache: fast GPU (hot) → slow GPU (warm) → CPU (cold)**.
- **Online + reactive** (no forecast): promote / demote / migrate / replicate experts as the hot
  set drifts, from measured recent load.
- Headline lever: **replicate-and-split** the hottest expert across tiers (split its tokens by
  tier speed) so neither tier bottlenecks.

## Ideas borrowed — paper references
- **Reactive cache mechanism** — Expert Buffering ([[Toward-Efficient-MoE-Inference]], NeurIPS'24): keep recently-hot experts resident. We extend its GPU↔CPU memory cache with a slow GPU tier.
- **Tier-aware placement + straggler objective + consistent/temporal taxonomy** — [[GEM]]
  (2605.19945) move from static to online.
- **Static optimum + all-to-all comm scheduling** — [[Aurora]] (2410.17043). 
- **Dynamic rebalance + replication substrate** — **SGLang EPLB** (DeepSeek). But it's **homogeneous** (balances token *count*, no tier-speed term), **coarse-cadence** (~every 1000 reqs), even-split replication. → it's the **homogeneous-dynamic baseline** *and* the **build substrate** (already has  replication + periodic rebalance + async overlap + disk staging); we add tier-awareness on top.

## Models — does the cache survive every MoE type?
| Model                        | Experts        | Role                                              |
| ---------------------------- | -------------- | ------------------------------------------------- |
| **Mixtral-8×7B**             | 8 (coarse)     | sharpest skew — "the cache pays"                  |
| **OLMoE-1B-7B**              | 64 (mid)       | bring-up + B2 headline (done)                     |
| **Qwen3-30B-A3B / DeepSeek** | 128–256 (fine) | stress test — does skew survive ~uniform routing? |

Metrics: imbalance-over-time, reactive hit-rate, temporal-mass (premise) → then **real p99 /
$/token / throughput** on H200 (fast) + RTX 6000 Ada (slow) end-to-end serving.

## Can we overlap expert migration with the all-to-all (EP)?
- EP already runs an **all-to-all every MoE layer** (dispatch → expert-GPUs → combine) over the
  interconnect; **migration rides the same link → contention.**
- **Bet:** migration is small + infrequent (few experts/window; hot set drifts ~6 s; one expert ≈
  12.6 MB ≈ 0.5 ms over PCIe) → it **hides under** the all-to-all.
- **Two ways it breaks:** 
- (1) migration steals bandwidth from the latency-critical dispatch/combine;
  (2) a **replicated** expert's tokens must be *split* in the all-to-all, which stock EP doesn't do.
- **In the offline/batch setting this gets much easier:** with many batches in flight, overlap the
  all-to-all *and* the migration with **other batches' compute**, the slow comm hides.
- **Precedent:** SGLang's **async EPLB rebalance already moves expert weights in a background thread, overlapping with compute** — so the mechanism exists; we extend it (tier-aware, finer cadence).
- **Plan:** schedule movement to overlap; reuse Aurora-style comm scheduling if contention bites; validate live (measured straggler vs predicted) — the key open systems question.

## Concerns
- **Tier gap must be measured right:** end-to-end serving / decode (bandwidth-bound), **not** an
  isolated expert-FFN (too small → looks like ~1.0). H200 vs Ada ≈ 3–5× real.
- **Fine-grained MoE may be ~uniform** (GEM: ~1.5% on Qwen3-128) → little to cache. Fallbacks:
  degrade to GEM-style tier-split (needs no skew); pin DeepSeek's always-on **shared expert**.
- **Only moderate concentration:** top-8 of 64 ≈ 36% of load → cache targets the **straggler**
  (busiest expert), not total volume; the cold tail rides the slow tier fine.
- **Cross-node EP:** H200 + Ada are on different nodes → all-to-all + migration over **Ethernet**
  (slow) → bad for *online* latency. Resolved by targeting offline generation (below); cleaner
  single-node options: 308 (Ada+A6000 in one box) or clock-cap a same-GPU pair.

## Offline / batch generation
- Hetero clusters → EP **across nodes** → all-to-all **slow over Ethernet** → kills *online* latency.
  Offline gen is **latency-relaxed** → cross-node OK; objective = **throughput / $**.
- *Sorting by domain* only flattens cross-domain drift (needs labels; within-domain ~66% temporal skew remains) → offline win = **minimize + hide the cross-node all-to-all**, not chase drift.
- **Contribution:** tier-aware placement + replication that **minimizes & hides the cross-node
  all-to-all** for throughput/$.
- **Tension:** cross-node EP is forced only for *large* MoE → fine-grained → ~uniform → least
  skew. The regime needing comm-hiding is where the cache premise is weakest — find a model where both hold, or split the claim (skew-cache for fit-one-node models; comm-hiding for large).
