# TierShift — the cache argument & how it works

Framing the online expert-cache for [[Research Plan 2]] (TierShift), grounded in the B2
measurements ([[first_benchmark_results]]) and the two anchor papers:
**Expert Buffering** ([[Toward-Efficient-MoE-Inference]], NeurIPS'24) and **[[GEM]]**.

---

## 0. The framing in one breath

**TierShift = Expert Buffering's reactive cache, lifted onto GEM's heterogeneous compute tiers,
and made dynamic.**

- **Expert Buffering** gives the *mechanism*: a **reactive cache** of recently-hot experts (react,
  don't predict) — but its cold tier is **CPU storage**, so a miss is a *stall to fetch*.
- **GEM** gives the *setting + objective*: **heterogeneous GPUs**, the **straggler at the all-to-all
  barrier**, and the **consistent/temporal** expert split — but its expert→GPU map is **static**.
- **TierShift** runs EB's reactive cache on GEM's heterogeneous **compute** tiers, online: **pin**
  the consistent experts, **migrate/cache** the temporal ones as they heat up, **replicate-split**
  the straggler. The cold tier is a **slow GPU that still computes**, so a miss is a *straggler, not
  a stall* — which is what makes migration an optimization rather than a correctness requirement.

So we take *what to keep resident* from EB, *where to put it and what to optimize* from GEM, and add
the *dynamic* piece (online migration + replicate-split) that neither has.

## 0a. Why model architecture (# experts) decides whether this works

The cache only pays if there's a **hot set** — and that's set by **routing granularity**, an
architecture property:

- **Coarse (few, big experts):** Mixtral-8×7B (8, top-2), OLMoE-64 (top-8) → experts specialize →
  **sharp, drifting skew** → the cache pays. (B2 confirms on OLMoE: imbalance 2.3–8.0.)
- **Fine-grained (many, small experts):** Qwen3-30B-A3B (128), DeepSeek (256) → load spreads
  ~uniformly → **little to cache** (GEM got ~1.5% there).

Granularity also sets the *costs*, not just the opportunity: # experts × expert size = the
**migration bytes** to move; # experts × top-k = the **all-to-all volume**. Finer routing → more
uniform (less to gain) *and* more, smaller experts (cheaper per-migration but more of them, more
all-to-all messages). So architecture determines **both** whether a hot set exists **and** the price
of chasing it — which is exactly why the **granularity sweep** (Mixtral → OLMoE → Qwen3) is the
scope test, not optional color. Where the cache stops paying, TierShift degrades to GEM-style
tier-split (which doesn't need skew).

## 0b. The systems bet — dynamic movement *shares* the all-to-all bandwidth

The thing we're betting "just works": in EP serving, **every MoE layer already does an all-to-all**
— dispatch each token to whichever GPU holds its expert, then combine results back — over the
interconnect (here **PCIe Gen4, no NVLink**). Our dynamic movement (**migrating ~12.6 MB experts,
replicating hot ones**) rides the **same interconnect**, so **migration traffic contends with the
all-to-all token traffic** for bandwidth.

The hope: migration is **infrequent and small** relative to per-window token volume (and the hot set
drifts on ~6 s, so we move few experts per window), so it **overlaps/hides under** the all-to-all
without materially hurting dispatch/combine — i.e. **EP + online expert movement composes
cleanly.** Two reasons it might *not*:
- migration bytes steal PCIe bandwidth from the latency-critical dispatch/combine in that window;
- **replication changes the all-to-all pattern** — a token for a replicated expert can now go to
  *either* tier, so the dispatch has to *split* that expert's tokens, which the stock EP all-to-all
  doesn't do out of the box.

Our **offline sim currently assumes migration cost is additive and ignores all-to-all contention** —
a known gap. This is exactly why **B7 (live cross-tier EP serving) is a required gate**: serve OLMoE
EP across both tiers with experts actually moving, and confirm the measured straggler/p99 matches the
offline model. Until B7, "EP + dynamic movement works" is an assumption, not a result.

## 1. The setup — what we're caching, and onto what

A mixed GPU fleet: a few **fast/expensive** GPUs (Ada/L40S/H200, FP8) + several **slow/cheap**
ones (A6000, Ampere). The MoE's experts must be placed across both tiers; at the all-to-all
barrier, **the slowest tier to finish its experts caps per-layer latency** (the straggler — GEM's
framing). The cache = **which experts live resident on the fast tier right now.**

Goal (the objective B2 set up): **hold a p99 latency SLO at minimum $/token** on this fleet —
i.e. don't pay for an all-fast deployment if a cheap tier + smart placement holds the SLO.

## 2. What we borrow (and re-validated on OLMoE)

**From Expert Buffering (the NeurIPS paper) — the *reactive cache* principle (our Layer 0):**
- Keep **recently-hot** experts resident; **react to measured recent load, don't forecast.** This
  is the answer to "you can't predict the workload" — you don't, you cache.
- B2 re-validates this on OLMoE: last window's hot set covers **83%** of the next window's (q5).
  So a reactive cache is enough — no model, workload-agnostic.

**From GEM — two gifts:**
- **The straggler objective** at the all-to-all barrier (faster tier should take proportionally
  more load so all tiers finish together).
- **The consistent/temporal taxonomy** → our *split strategy*: B2 shows ~34 experts are
  consistent and **66% of hot-expert load is temporal** (bursty). GEM itself names the wedge:
  *"the static mapping does not account for distribution shift in token routing post-deployment."*
  TierShift **is** that adaptation.

## 3. What's actually new (so it's not "Expert Buffering ported to 2 GPUs")

The carve-out, from the paper notes — these are the contribution, the rest is borrowed:

| | Expert Buffering / GEM | **TierShift** |
|---|---|---|
| Cold tier | CPU memory (storage) / one-time map | **a slow GPU that still computes** |
| A "miss" | a **stall** to fetch weights | **a straggler** (served slower, never stalls) |
| Objective | hit-rate / memory saved / latency | **makespan-straggler + $/token at p99 SLO** |
| Hot expert too big for 1 GPU | — | **replicate across tiers + split tokens by speed** |
| Adaptation | static (GEM) / memory-LRU (EB) | **online migration as the hot set drifts** |

The single most important new lever: **replicate-and-split.** When one expert is hotter than a
fast GPU's share (the straggler), put a *replica* on a second tier and **split its tokens in
proportion to tier speed** so both finish together. Neither anchor does this (EB's CPU copy is a
backup, not a load-sharing replica; GEM is one-expert-one-GPU).

## 4. How it works — the control loop (per ~1 s window)

Free input: the router already tells us per-expert load every forward (our B2 hook). Each window:

1. **Measure** recent per-(layer, expert) load (last window, or a short EMA — B2 says a 2–3
   window memory lifts the hit-rate 0.83 → 0.91).
2. **Pin the consistent experts** on the fast tier (the ~13% of load that's always-hot — set once,
   never evict). Cheap and stable.
3. **Reactively cache the temporal experts:** admit the newly-hot ones onto the fast tier, evict
   the least-recently-hot to make room (LRU-style). This chases the 66% temporal mass.
   - *Feasible because* migration ≈ **0.5 ms** ≪ hot-set turnover ≈ **6 s** (q3, ~12,000× margin)
     → there's time to move an expert before the hot set moves on.
4. **Replicate-and-split the straggler:** if the hottest resident expert exceeds a fast GPU's
   share, place a replica on the slow tier and split its tokens by the measured tier speed ratio
   (the fast:slow tok/s from the tier bench) so both tiers finish together.
5. **Leave the cold tail on the slow tier.** It's 36% of *aggregate* load but **light per-expert**
   — no single cold expert is a straggler, so the slow tier's collective throughput absorbs it.
   (Important nuance from B2: top-8 capture only ~36% of load; the cache targets the *straggler*,
   not total volume.)

Sizing: the fast-tier capacity (how many experts resident) and the fast:slow GPU mix are the
$/token knobs — pick the smallest/cheapest fleet that still holds p99.

## 5. Why B2 says this can work (and the honest caveat)

- **Drift is real & cacheable:** imbalance 2.3→8.0, domain-distinct, reactive hit-rate 0.83 (q1/q5).
- **Migration ≪ drift:** ~12,000× margin → online adaptation is feasible (q3).
- **Headroom over static exists:** 66% of hot-expert load is temporal → a static map (GEM) leaves
  most of it on the table; that's the ceiling TierShift chases (q6 / taxonomy).
- **Caveat (don't oversell):** routing is only *moderately* concentrated (top-8 ≈ 36% of load), and
  the real **fast/slow tier gap is unmeasured** (modern cards showed ~1.0; needs the A6000). So the
  *latency/$ payoff* is still pending the tier-gap run — B2 proves the *premise*, not the payoff.

## 5a. Comparing against Aurora (the static optimum — the hardest boundary)

[[Aurora]] is the most dangerous "static core" neighbor — *more* dangerous than GEM, because GEM is
a heuristic but **Aurora is the static *optimum***: it jointly solves **expert placement + all-to-all
comm scheduling**, proves optimality in 3 of 4 cluster regimes, approximates the NP-hard
colocated-on-heterogeneous case to a **1.07× gap**, and reports **3.54× hetero speedup**. So "beat
Aurora" means **beat the best possible static plan**, not a heuristic — a strong bar.

**Why TierShift still has a wedge (all from Aurora's own text):**
- **Static, from a-priori statistics.** Aurora optimizes from *"historical statistics known a
  priori"* — no migration, no replication-for-hotness, no online re-placement. Its only nod to
  non-stationarity is *noise applied after deployment* (15.8% degradation at 75% noise) — **noise,
  not structured drift.** Our drift is structured (domain shifts) and larger.
- **Its gap list is our thesis:** *"online dynamic rebalancing for temporal workload variation."*
- **Theory + simulation, no real hardware / no serving stack.** We run on real GPUs (B2/B7).

**The honest positioning** (don't overclaim): *we do not beat Aurora at static placement — we show
the static optimum itself mis-fits a drifting workload, which is exactly the case Aurora excludes.*

**Aurora's comm-scheduling half is *composable*, not a competitor.** Aurora co-optimizes the
all-to-all *schedule* — the very thing our §0b "bandwidth bet" hand-waves. So Aurora is two halves:
(i) static placement = the baseline we beat under drift; (ii) all-to-all scheduling = a component we
could **adopt** to make our dynamic movement actually hide under the all-to-all. Frame (ii) as
orthogonal/reusable, not as something we're competing with.

**How we compare experimentally** (we can't run their code — it's theory/sim, no integration):
- Implement **`aurora-static-optimal`** as a sim config = the **best static** placement given the
  *whole-trace* average expert load + per-tier speed (speed-proportional hot→fast, their model). It
  is the strongest static baseline — strictly better than our current `static-avg-opt`/`GEM-style`,
  so make it the offline optimum, computed with full knowledge of the average.
- Report the **Aurora→TierShift gap** next to the GEM gap and static-avg→oracle gap. The win must
  come from the **temporal mass** (66%) that *any* static average — even the optimal one — cannot
  place. If that gap ≈ 0, Aurora suffices and there's no contribution.
- Be explicit in the writeup: "Aurora-static-optimal" reproduces their *strategy* as a config, not
  their system; we cite their 3.54× as the static ceiling and measure how far drift pushes below it.

## 6. The baselines TierShift must beat (or it's not a contribution)

- **Naive Expert-Buffering port** (LRU hot→fast, no split) = **Layer-1 baseline**. If
  reactive+split ≈ this, TierShift is "just EB on two GPUs" → no contribution. *(B2's
  make-or-break gap, currently ~0 because tier ratio ≈ 1.0 — needs the A6000 to test.)*
- **GEM-style static speed-aware map** = the dangerous *heuristic* neighbor. Win must come from the
  **temporal** experts (66%) a static map can't place — the GEM gap.
- **Aurora-static-optimal** = the dangerous *optimal* neighbor (the **best possible** static plan +
  comm schedule). The strongest static bar; the same temporal-mass argument applies, and it's the
  ceiling our "online beats static" claim is measured against (see §5a).
- **All-fast deployment** = the cost reference: must hold the same p99 at lower $/token.

## 7. Open design questions (for the controller phase, M1–M4)

- **Cache size vs. tail:** since top-8 is only ~36% of load, how big must the resident set be to
  hold p99? (sweep `cap-fast-frac`.)
- **Hysteresis:** evict on K cold windows, not 1 (B2: 3-window memory → 0.91 hit-rate) — avoid
  thrashing at domain boundaries.
- **Migration accounting:** real PCIe ms (not the analytic estimate) and whether to migrate vs.
  replicate-only when the gap is small (the Q3 regime map).
- **Phase-awareness:** decode is the latency-critical, bandwidth-bound phase — does the hot set /
  cache policy differ for decode vs prefill? (needs the per-token phase split.)
- **Granularity:** does this survive fine-grained MoE (Qwen3-30B-A3B, ~uniform routing → little to
  cache) — GEM's warning. Coarse models (OLMoE, Mixtral) are where the cache pays.
