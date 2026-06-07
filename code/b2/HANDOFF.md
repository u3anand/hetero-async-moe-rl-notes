# B2 Handoff — start here

Entry point for the next agent picking up **B2 (TierShift)**. Read this, then the canonical docs, then start on the critical path. B2 is a *new, separate* direction from B1 (AutoPD-RL) — different code tree (`code/b2/`), serving-only, no RL.

## TL;DR — what B2 is
Serve a fixed **MoE** model (OLMoE) on the heterogeneous **watgpu308** node, replay a **mixed drifting request trace**, **log which experts fire over time**, and answer three go/no-go questions that decide whether the "TierShift" system is worth building. It's a **characterization benchmark** — a cheap kill-test for our own idea — not the system itself.

**The idea being tested:** on a mixed fleet (cheap slow GPUs + a few fast ones), reactively **cache hot MoE experts on the fast tier** (and replicate/split the hottest) to **hold a p99 latency SLO at lower $/token** than an all-fast-GPU deployment — adapting as the hot set drifts with the workload.

## Read-first (in order)
1. `Research Plan 2.md` (vault root) — the strategy: motivation, the neighbor/novelty map, the **layered novelty** (§3), the **objective** (p99 SLO at min $/token).
2. `Initial Plan 2.md` (vault root) — the gated B2 plan (steps 0–7, Q1–Q7, instrumentation, definition of done).
3. `code/b2/RUNBOOK.md` — the ordered commands (Track A laptop + Track B cluster).
4. `Benchmark Comparison.md` (vault root) — B1 vs B2 side by side (why B2 is ~⅓ the lift).
5. Anchor papers (`papers/`): [[Toward-Efficient-MoE-Inference]] (Expert Buffering = the reactive-cache idea we build on), [[Aurora]] + [[MegaScale-Infer]] (static hetero placement — the boundary), [[SwapMoE]].

## What success means — the 3 questions B2 must answer
1. **Locality (Q1/Q5):** does the hot expert set stay put long enough to cache? (flat/random → project dies here.)
2. **Kill-test (Q3):** can an expert migrate over PCIe (~1 ms) faster than the hot set churns (~seconds)? (no → fall back to replication-only.)
3. **Make-or-break (Q4):** does the **reactive+split** policy beat the **naive LRU-port** (= "just copy Expert Buffering to two GPUs")? (≈ → no contribution; must detect early.)

Plus the **objective metrics** layered on the offline sim: **p99 latency / SLO-attainment** and **$/token** per placement config (not just straggler/throughput).

## Decisions already locked (don't re-litigate)
- **Serving only** — no RL, no trainer, no udocker reward. That's B1.
- **BF16 for the whole OLMoE phase.** The slow tier (A6000, Ampere) has *no* FP8 hardware, and routing/hot-set capture is precision-independent. FP8 is measured *only* as a tier-gap micro-bench (FP8-Ada vs BF16-Ampere widens the gap ~2.4×→~4×, which helps the thesis). Full FP8 serving waits for the Qwen3-30B-A3B phase.
- **vLLM**, not SGLang — the repo already forks it (`code/forks/vllm`) and the expert-logging hook is **the same one B1-A2 needs**, so it's shared, not wasted. Revisit SGLang only at the real cross-tier-EP-*serving* phase (its EPLB is the comparison point then).
- **Reactive cache, NOT workload prediction.** The controller reacts to measured recent expert load (free from the router). No forecasting. This is what makes it workload-agnostic / usable as a reference.
- **Most of B2 is single-GPU capture + 2 micro-benches + offline simulation.** Live cross-tier EP serving (B7) is the *last, optional* validation gate — NOT a prerequisite for the headline charts. (Routing is hardware-independent, so capture on one GPU is valid.)
- **Objective = hold p99 SLO at min $/token on a mixed fleet** (cost-first), with straggler/throughput as the mechanism beneath it.

## Critical path (the bottleneck is code, not the node)
B2 has plans + runbook + scaffolding, but **no implementation yet**. Ordered:

0. **Git:** create branch `b2-tiershift` off `main`; commit the B2 deliverables (file map below). Author as `u3anand`, no agent trailers (see `CLAUDE.md`). Nothing is committed yet.
1. **Track A — the vLLM MoE-gate hook (laptop, the keystone).** In `code/forks/vllm`, find the OLMoE MoE block (after `router_logits, _ = self.gate(...)` + top-k select), increment a per-`(layer_id, expert_id)` counter into a ring buffer; a sampler flushes windowed rows to `expert.jsonl`. Sampling-rate gated, `VLLM_USE_PRECOMPILED=1`, no kernel changes. **Shared with B1-A2.** Push, bump `pins.txt`.
2. **`code/b2/setup.sh`** — conda env + vLLM fork install + OLMoE download (simpler than B1: no udocker/prime-rl).
3. **`code/b2/src/b2tel/`** — `telemetry.py` (expert.jsonl writer), `make_trace.py` (mixed chat/code/math drifting trace), `replay.py` (replay vs a vLLM endpoint), `tier_bench.py` (Ada vs A6000 expert-FFN tokens/s → `tier_cost.json`), `migrate_bench.py` (PCIe expert-weight transfer ms + hot-set turnover → `migration_cost.json`), `simulate.py` (join load×tier×migration×`fleet_cost.json`; score configs on **p99 latency + $/token**), `charts.py` (Q1–Q7). Plus `fleet_cost.json` ($/hr per tier).
4. **Cluster (watgpu308):** B0 recon → B1 env → B2 trace → **B3 capture** (Q1) → **B4 tier-bench** → **B5 migrate-bench** (Q3) → **B6 sim+charts** (Q2/Q4). See RUNBOOK.

**Earliest kill-signals to prioritize:** B5 migration-ms (no hook needed) + a quick B3 capture → answers Q3 and Q1 fastest. The make-or-break Q4 comes from B6.

## Cluster access + status
- SSH: `ssh u3anand@watgpu.cs.uwaterloo.ca` (key-based, BatchMode-ok). Login node has no GPU — never compute on it.
- Reliable lane: **watgpu308** (`-p SCHOOL`), 4×L40S + 2×A6000 + 2×RTX6000 Ada, PCIe Gen4, **no NVLink**.
- **Always check first:** `ssh u3anand@watgpu.cs.uwaterloo.ca 'bash -s -- watgpu308' < code/b2/slurm/check-node.sh`
- **Status 2026-06-07 ~17:39Z: watgpu308 READY (8/8 free).**
- Gotchas (from `WatGPU.md`, local-only/gitignored): set `NCCL_SOCKET_IFNAME=eth0` (multi-GPU only); pin tiers by CUDA index via `nvidia-smi -L`; jobs are preemptible+requeued → checkpoint; build inside an interactive job, not on login (NFS home is noexec).

## File map (the B2 deliverables to commit)
- Vault: `Research Plan 2.md`, `Initial Plan 2.md`, `Benchmark Comparison.md`
- `papers/`: `Aurora.md`, `MegaScale-Infer.md`, `HeterMoE.md`, `Helix.md`, `HexGen-2.md`, `Toward-Efficient-MoE-Inference.md`, `SwapMoE.md` (+ `Index.md`, `_index.tsv` updates; PDFs are gitignored)
- `code/b2/`: `README.md`, `RUNBOOK.md`, `HANDOFF.md` (this), `slurm/check-node.sh`, dirs `src/b2tel/`, `traces/` (to fill)
- `.gitignore` — extended for `code/b2/{runs,models,traces}` + `*.jsonl`

## What to NOT do
- Don't serve in FP8 on OLMoE (slow tier can't; not needed). Don't switch to SGLang yet. Don't build the controller (that's the *next* phase — B2 only profiles). Don't add CPU-offload (deliberately out of scope — it's the crowded cell). Don't try to "predict the workload" — react to measured load.
