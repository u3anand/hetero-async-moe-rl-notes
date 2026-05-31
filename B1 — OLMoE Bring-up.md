# B1 — OLMoE-1B-7B Bring-up (3 configs) on watgpu308

The first runnable milestone of [[Research Plan]] §B1. Goal: get the **full async MoE-RL loop running end-to-end on one node** with **OLMoE-1B-7B**, run it under the three baseline configs (**homo-fast / homo-slow / hetero**), and emit the Q1–Q7 telemetry on this small model. This is "make it work" + the 3-config comparison — not scale. Hardware and cluster constraints: [[WatGPU]].

OLMoE-1B-7B is ~7B (~14 GB BF16) → **fits one 46 GB card → no tensor-parallel, no FP8 anywhere this phase** — simplest possible plumbing while still exercising the MoE code paths (expert-parallel, router-logit exposure for Q6, router/expert broadcast for Q4), so the later Qwen3-30B-A3B swap is "just bigger."

## The 3 configs — fixed budget, vary only the rollout tier

Controlled comparison: **train held constant; rollout replica *count* held constant (2); only the rollout *tier* changes.** Isolates the Ada-vs-Ampere tier effect.

| Config | Rollout pool `D_I` (2 replicas, 1 GPU each) | tier-mix point |
|---|---|---|
| **homo-fast** | 2× Ada (e.g. 2× RTX 6000 Ada) | 100/0 |
| **homo-slow** | 2× RTX A6000 | 0/100 |
| **hetero** | 1× Ada + 1× A6000 | 50/50 |

- **Training pool `D_T` (all 3 configs): 2× L40S**, FSDP2+LoRA. Held constant as a control (on Ada — naive-hetero design: homogeneous train, hetero rollout; avoids FSDP-on-Ampere). Model fits 1 card; we shard across 2 only to exercise the FSDP2 path the 30B will need.
- Each config uses **4 GPUs** (2 train + 2 rollout); **run configs sequentially** so each owns all 32 CPU cores (running two at once would confound throughput via CPU contention). Grab the node `--exclusive`.
- Verify index→model with `nvidia-smi -L` on first alloc; pin every process via `CUDA_VISIBLE_DEVICES` (fall back to GPU-UUID pinning if ordering drifts).

## GPU/CPU budget on watgpu308
8× `schoolgpu` = 4 L40S + 2 RTX A6000 + 2 RTX 6000 Ada (46–48 GB, PCIe Gen4, no NVLink). **Only 32 CPU cores + 817 GB RAM** for 8 GPUs — the udocker reward sandboxes share these cores (no separate CPU pool), so **CPU is the likely rollout bottleneck** and sandbox-queue-depth/CPU-saturation is itself the Q2 measurement.

## Components to reuse (don't reinvent)
**prime-rl** (orchestrator / trainer / inference-service; GRPO, bounded-staleness async, FSDP2+LoRA, vLLM, DTensor EP) · **mini-swe-agent** (bash-only stateless loop; wrap exec in udocker) · **udocker** R2 (runs `ghcr.io/epoch-research/swe-bench.eval.x86_64.*` images for the real-pytest reward) · dataset `nebius/SWE-rebench-openhands-trajectories` (2 GB parquet).

## Dir layout (`/u3/u3anand/b1/`)
`repos/{prime-rl,mini-swe-agent}` · `models/` (HF_HOME, OLMoE) · `data/` (SWE-rebench parquet) · `sandbox/.udocker` · `runs/<config>/<ts>/{ckpt,logs,telemetry}` · `charts/` · `slurm/`. Global env in every job: `NCCL_SOCKET_IFNAME=eth0`, `HF_HOME`, `UDOCKER_DIR`.

## Steps (each gated GO/NO-GO)

**0 — Recon** (`salloc -p SCHOOL -w watgpu308 --gres=gpu:1`): `nvidia-smi -L` → index→tier map; confirm `eth0`; check node-local scratch. **Gate:** map = {6 Ada, 2 A6000}; eth0 confirmed.

**1 — Env** (`salloc --gres=gpu:2 -t 7:00:00 --cpus-per-task=16`): `conda create -n b1 python=3.11` + `conda install -c nvidia cuda=12.4`; install prime-rl (uses `uv`) + mini-swe-agent; `pip install --user udocker && udocker install`; download OLMoE + dataset in background. **Gate:** torch/vLLM/flash-attn import; udocker runs alpine; 2-GPU NCCL all-reduce no hang.

**2 — Sandbox spike** (CPU/udocker, ∥ step 3): pull ~15 epoch-research SWE-bench images; `udocker setup --execmode=R2`; run each repo's pytest; log `pull_s, setup_s, test_wall_s, cores_used, exit_code, ok`. → seeds **Q1/Q5** anchors **and the concurrency cap (cores-per-test)**. **Gate:** ≥80% clean pass/fail; cores-per-test measured. (Fallback: exec mode P2; reuse containers; stage UDOCKER_DIR to local scratch.)

**3 — Inference spike** (GPU): vLLM serve **OLMoE single-card on 1 Ada (BF16)**; drive ~100-turn long context → validate KV fit + **router-logit exposure (Q6 hook)** + ~1.5 s/turn anchor; then serve on **1 A6000** → record **fast:slow tokens/s ratio** (the tier gap the 3 configs will exhibit). **Gate:** serves on both tiers; ratio recorded; router logits extractable.

**4 — Train spike** (GPU, `D_T` = 2 L40S): prime-rl **FSDP2+LoRA single GRPO step** on OLMoE; **write + resume a checkpoint** (the preemption primitive). Record C_train baseline. **Gate:** step + ckpt-resume work.

**5 — Integration + run the 3 configs** (`sbatch -w watgpu308 --exclusive --requeue --signal=B:SIGUSR1@180`):
   - **5a.** Bring the loop up on **homo-fast first** (all-Ada, no cross-tier): wire prime-rl's 3 services to `D_T`=2 L40S + `D_I`=2 Ada; mini-swe-agent reward via udocker; SIGUSR1→ckpt-trap→resume. Run to ~30–60 min **steady state** (queue non-empty, staleness bounded, real rewards). **Gate 5a:** steady state + survive simulated preempt.
   - **5b.** Then run **homo-slow** (`D_I`→2 A6000) and **hetero** (`D_I`→1 Ada + 1 A6000) by changing only the rollout `CUDA_VISIBLE_DEVICES` pinning. Sequential runs, each owning all 32 cores. **Gate 5b:** all 3 configs reach steady state and log telemetry.

**6 — Instrumentation + comparison charts** (Q1–Q7 → JSONL → charts): hook each question (Q1 categorize+time bash cmds; Q2 GPU busy% + sandbox queue depth; Q3 C_I vs C_T; Q4 router-vs-expert broadcast bytes/time; Q5 per-episode wallclock CDF; Q6 cross-replica router-KL via per-token logits+version+replica_id; Q7 MFU + HBM-bw). Charts plot **homo-fast vs homo-slow vs hetero side by side**. **Gate:** every hook emits; all charts render across the 3 configs. *Note: under LoRA the Q4 broadcast is just adapter deltas → Q4 is a smoke-test here (real Q4 needs the 30B/full-FT phase). Q6 fallback: subset-sample logits or recompute trainer-side.*

## Definition of done
On watgpu308, the OLMoE-1B-7B async GRPO loop runs to a steady-state window under **all three configs** with **real udocker test-suite rewards**, **survives a preemption/requeue**, and emits Q1–Q7 telemetry that renders to **side-by-side comparison charts**. The fast:slow throughput gap and where hetero lands between the homogeneous baselines are quantified on the small model. Anchor numbers (turns/episode, inference/turn, bimodal bash latency, mid-episode test count, episode CDF) recorded. This proves the harness; the Qwen3-30B-A3B phase reuses it unchanged.

## Out of scope this phase (deferred)
- Qwen3-30B-A3B / FP8 / TP=2 (next phase, harness unchanged).
- Finer tier-mix points **75/25 and 25/75** — need ≥4 rollout replicas; this phase does the 3 endpoints at 2 replicas.
- 16-GPU / multi-node; cluster-size sweep; M4 (only *measured* via Q2, not built).

## Top risks → fallbacks
Preempt → SIGUSR1 ckpt trap + `--requeue` (validated steps 4–5). · NCCL hang → `NCCL_SOCKET_IFNAME=eth0` (+`NCCL_P2P_DISABLE=1`). · GPU index drift → UUID pin. · **CPU saturation from sandboxes** → cap concurrency to cores-per-test (also the Q2 finding); run configs sequentially. · udocker NFS I/O → local scratch + reuse containers + R2→P2. · LoRA makes Q4 toy → note, defer real Q4 to 30B/full-FT.

## Sequencing
Spine: 0 → 1 → (2, 3, 4) → 5a → 5b → 6. Parallel: step 2 (CPU) ∥ step 3 (GPU); model/dataset downloads background from step 1; step-6 hook design overlaps 5a stabilization. The 3 configs run sequentially (shared 32-core CPU).
