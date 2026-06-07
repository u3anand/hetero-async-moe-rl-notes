# B1 — OLMoE-1B-7B Bring-up + PD-Demand Profiling on watgpu308

The first runnable milestone of [[Research Plan]] §B1 (AutoPD-RL). Goal: get the **full async agentic-RL rollout loop running end-to-end on one node** with **OLMoE-1B-7B**, with **prefill and decode disaggregated** across heterogeneous tiers, run it under **static-PD baseline configs**, and emit the Q1–Q7 PD-demand telemetry that motivates the online controller (B2). This is "make it work" + the static-PD comparison — not scale. Hardware/cluster constraints: [[WatGPU]].

OLMoE-1B-7B is ~7B (~14 GB BF16) → **fits one 46 GB card → no tensor-parallel, no FP8 this phase** — simplest plumbing while still exercising disaggregated prefill/decode vLLM instances and the async rollout→trainer loop, so the later Qwen3-30B-A3B swap is "just bigger." (The model being MoE is now incidental — the thesis is PD allocation, not MoE internals.)

## The static-PD configs — fixed hardware, vary only the prefill/decode split

Controlled comparison: **train held constant; total rollout GPUs held constant (4); only the static prefill:decode instance assignment changes.** The point is to show **no single static split wins across workload mixes** and that PD demand is non-stationary within a run — the motivation for an automatic controller.

| Config | Prefill pool | Decode pool | tier-affinity point |
|---|---|---|---|
| **prefill-heavy** | 2× Ada + 1× A6000 | 1× A6000 | favors context ingest |
| **decode-heavy** | 1× Ada | 1× Ada + 2× A6000 | favors token generation |
| **balanced** | 1× Ada + 1× A6000 | 1× Ada + 1× A6000 | even split |

- **Training pool `D_T` (all 3 configs): 2× L40S**, FSDP2+LoRA. Held constant as a control. Model fits 1 card; shard across 2 only to exercise the FSDP2 path the 30B needs.
- Prefill is compute-bound → prefer fast **Ada** tier; decode is bandwidth-bound → tolerates **Ampere**. Each config uses **4 rollout GPUs + 2 train GPUs = 6**; **run configs sequentially** so each owns all 32 CPU cores (concurrent runs confound throughput via CPU contention). Grab the node `--exclusive`.
- Verify index→model with `nvidia-smi -L` on first alloc; pin every process via `CUDA_VISIBLE_DEVICES` (UUID-pin fallback if ordering drifts).

## GPU/CPU budget on watgpu308
8× `schoolgpu` = 4 L40S + 2 RTX A6000 + 2 RTX 6000 Ada (46–48 GB, PCIe Gen4, no NVLink). **Only 32 CPU cores + 817 GB RAM** for 8 GPUs — the udocker reward sandboxes share these cores, so **CPU is a likely rollout bottleneck** and sandbox-queue-depth/CPU-saturation is itself a Q1/Q3 measurement (env-heavy phases).

## Components to reuse (don't reinvent)
**prime-rl** (orchestrator / trainer / inference-service; GRPO, bounded-staleness async, FSDP2+LoRA, vLLM, SampleBuffer) — **wire vLLM in prefill/decode-disaggregated mode** · **mini-swe-agent** (bash-only stateless loop; wrap exec in udocker) · **udocker** R2 (runs `ghcr.io/epoch-research/swe-bench.eval.x86_64.*` images for the real-pytest reward) · dataset `nebius/SWE-rebench-openhands-trajectories` (2 GB parquet).

## Dir layout (`/u3/u3anand/b1/`)
`repos/{prime-rl,mini-swe-agent}` · `models/` (HF_HOME, OLMoE) · `data/` (SWE-rebench parquet) · `sandbox/.udocker` · `runs/<config>/<ts>/{ckpt,logs,telemetry}` · `charts/` · `slurm/`. Global env in every job: `NCCL_SOCKET_IFNAME=eth0`, `HF_HOME`, `UDOCKER_DIR`.

## Steps (each gated GO/NO-GO)

**0 — Recon** (`salloc -p SCHOOL -w watgpu308 --gres=gpu:1`): `nvidia-smi -L` → index→tier map; confirm `eth0`; check node-local scratch. **Gate:** map = {6 Ada, 2 A6000}; eth0 confirmed.

**1 — Env** (`salloc --gres=gpu:2 -t 7:00:00 --cpus-per-task=16`): `conda create -n b1 python=3.11` + `conda install -c nvidia cuda=12.4`; install prime-rl (uses `uv`) + mini-swe-agent; `pip install --user udocker && udocker install`; download OLMoE + dataset in background. **Gate:** torch/vLLM/flash-attn import; udocker runs alpine; 2-GPU NCCL all-reduce no hang.

**2 — Sandbox spike** (CPU/udocker, ∥ step 3): pull ~15 epoch-research SWE-bench images; `udocker setup --execmode=R2`; run each repo's pytest; log `pull_s, setup_s, test_wall_s, cores_used, exit_code, ok` **+ repo size / #files / dep files / test cmd** (the SWE features for Q6). → seeds **Q1/Q5/Q6** anchors **and the concurrency cap (cores-per-test)**. **Gate:** ≥80% clean pass/fail; cores-per-test measured. (Fallback: exec mode P2; reuse containers; stage UDOCKER_DIR to local scratch.)

**3 — PD inference spike** (GPU): vLLM serve **OLMoE in disaggregated prefill/decode mode** — a prefill instance on 1 Ada + a decode instance on 1 Ada (BF16); drive a ~100-turn long-context agentic episode → validate KV fit + **per-turn prefill/decode token split, TTFT, TPOT** (the Q2/Q4 hooks) + ~1.5 s/turn anchor; then move the decode instance to **1 A6000** → record **fast:slow decode tokens/s ratio** and prefill-vs-decode cost on each tier (the tier gap the static configs exhibit). **Gate:** disaggregated serve works on both tiers; per-turn P/D tokens + TTFT/TPOT extractable; ratio recorded.

**4 — Train spike** (GPU, `D_T` = 2 L40S): prime-rl **FSDP2+LoRA single GRPO step** on OLMoE; **write + resume a checkpoint** (the preemption primitive); confirm **SampleBuffer fill + `get_batch` blocking** is observable (the Q3 trainer-idle hook). Record C_train baseline. **Gate:** step + ckpt-resume work; SampleBuffer/idle instrumented.

**5 — Integration + run the static-PD configs** (`sbatch -w watgpu308 --exclusive --requeue --signal=B:SIGUSR1@180`):
   - **5a.** Bring the loop up on **balanced first**: wire prime-rl's services to `D_T`=2 L40S + disaggregated prefill/decode pools; mini-swe-agent reward via udocker; SIGUSR1→ckpt-trap→resume. Run to ~30–60 min **steady state** (queue non-empty, staleness bounded, real rewards). **Gate 5a:** steady state + survive simulated preempt.
   - **5b.** Then run **prefill-heavy** and **decode-heavy** by changing only the prefill/decode instance→GPU pinning. Sequential runs, each owning all 32 cores. **Gate 5b:** all 3 configs reach steady state and log telemetry.

**6 — Instrumentation + comparison charts** (Q1–Q7 → JSONL → charts): hook each question (below). Charts plot **prefill-heavy vs decode-heavy vs balanced side by side**, and — the headline — **P:D demand over time within each run** (Q2) against the fixed static splits, showing where each static split leaves prefill or decode idle. **Gate:** every hook emits; all charts render across the 3 configs.

## Instrumentation

Everything reduces to **three append-only JSONL streams** in `runs/<config>/<ts>/telemetry/`, joined on `episode_id / step_id / worker_id` and one monotonic clock (GPU phases timed with CUDA events). **Raw JSONL is the single source of truth; every chart is derived offline.**

**1. `episode.jsonl`** — one record per episode (prime-rl orchestrator + udocker env), with per-turn PD detail:
```
episode_id, worker_id, tier, policy_version_start, t_start, t_end,
num_turns, reward, staleness_at_consume,
swe_features = { repo_size, n_files, dep_files, test_cmd, docker_cache_hit, issue_len },
turns[] = { in_tokens, out_tokens, ttft_s, tpot_s, prefill_s, decode_s },
tool_calls[] = { category, t_start, t_end, dur_s, cores, in_udocker }
```
**2. `step.jsonl`** — one record per train update (trainer/orchestrator):
```
step_id, t_train, samplebuffer_fill, trainer_idle_s, get_batch_wait_s,
tokens, useful_tokens, batch_staleness[], stale_aborted_count
```
**3. `system.jsonl`** — ~1 Hz sidecar sampler, per worker type:
```
per-GPU sm_active%, dram_active% (HBM bw), worker_role{prefill|decode|train};
prefill_queue_depth, decode_queue_depth, sandbox_queue_depth, n_active_sandboxes, cpu_util
```

| Q | Hook (component) | Raw fields | Derived metric → chart |
|---|---|---|---|
| Q1 | mini-swe-agent udocker env: time+classify each bash cmd {file_op/git/mid_test/build/reward_eval} + tag prefill/decode time per turn | `turns[]`, `tool_calls[]` | seconds-by-phase (prefill/decode/env) → stacked bar |
| Q2 | inference svc: per-turn in/out tokens → P:D demand ratio over wall-clock | `turns[].in_tokens/out_tokens`, `prefill_s/decode_s` | **P:D ratio time-series vs static split → the headline non-stationarity chart** |
| Q3 | sampler (per-role `sm_active%`, queue depths) + trainer (`samplebuffer_fill`, `trainer_idle_s`) | system + step | GPU busy% by role + trainer idle vs PD split → dual-axis; **controller-win sizing** |
| Q4 | inference svc: TTFT + TPOT per turn, context growth | `turns[].ttft_s/tpot_s` | TTFT/TPOT + context-length trajectories → line (the predictable control signals) |
| Q5 | episode log `t_end−t_start` + per-episode phase mix | `episode.jsonl` | wallclock CDF p50/p95/p99 + per-episode P/D/env regime → CDF + regime scatter |
| Q6 | join `swe_features` → episode PD/env class label (prefill-heavy/decode-heavy/env-heavy/failure) | episode + sandbox spike | feature→class predictability (simple model AUC) → confusion/feature-importance; **M4 trigger** |
| Q7 | sampler `dram_active%` + prefill/decode queue depths | system | HBM-bw + queue depth per role → bar/series |

Reported **p50/p95/p99 over the steady-state window**, as a surface across configs (prefill-heavy/decode-heavy/balanced now; finer static splits + the online controller later) — never single means.

## Code & repo
Our own code (telemetry lib, sampler, charts, configs, slurm, sandbox helpers, `setup.sh`) lives in **`code/b1/`** in the notes repo. The three upstreams are **forked and owned** — `u3anand/{prime-rl, vllm, mini-swe-agent}` — git submodules under `code/forks/`, with **`pins.txt`** recording the tested commit triple. Fork edits: prime-rl (SampleBuffer-fill + trainer-idle + queue/staleness emits, **disaggregated prefill/decode wiring**, env→udocker), vLLM (per-turn prefill/decode token + TTFT/TPOT exposure; PD-disaggregation config — install `VLLM_USE_PRECOMPILED=1`), mini-swe-agent (`UdockerEnvironment` + tool timing + SWE-feature capture). Big artifacts (models, dataset, runs, JSONL, `.udocker`) stay on WatGPU at `/u3/u3anand/b1`, gitignored.

## Definition of done
On watgpu308, the OLMoE-1B-7B async GRPO rollout loop runs to a steady-state window under **all three static-PD configs** with **disaggregated prefill/decode**, **real udocker test-suite rewards**, **survives a preemption/requeue**, and emits Q1–Q7 telemetry rendering to **side-by-side comparison charts** — headlined by the **P:D-demand-over-time vs static-split** chart (Q2) and the **trainer-idle vs PD-split** chart (Q3). Anchor numbers recorded (turns/episode, TTFT/TPOT, per-turn P/D tokens, fast:slow decode ratio, episode CDF, SampleBuffer idle %). This proves the harness + motivates the controller; the Qwen3-30B-A3B phase reuses it unchanged.

## Out of scope this phase (deferred)
- Qwen3-30B-A3B / FP8 / TP=2 (next phase, harness unchanged).
- The **online PD controller itself** (B2) — this phase only *profiles* and shows static splits mis-fit; it does not yet re-size pools online.
- Finer static-split points; 16-GPU / multi-node; the SWE predictor M4 is only *measured* via Q6, not built into a controller.

## Top risks → fallbacks
Preempt → SIGUSR1 ckpt trap + `--requeue` (validated steps 4–5). · NCCL hang → `NCCL_SOCKET_IFNAME=eth0` (+`NCCL_P2P_DISABLE=1`). · GPU index drift → UUID pin. · **CPU saturation from sandboxes** → cap concurrency to cores-per-test (also a Q1/Q3 finding); run configs sequentially. · udocker NFS I/O → local scratch + reuse containers + R2→P2. · vLLM PD-disaggregation on a 1-card model is artificial at this scale → treat OLMoE PD numbers as harness-proving anchors; real PD pressure shows at the 30B phase. · KV-transfer cost between prefill/decode instances over PCIe (no NVLink) → measure it (Q7) as a real hetero-PD cost.

## Sequencing
Spine: 0 → 1 → (2, 3, 4) → 5a → 5b → 6. Parallel: step 2 (CPU) ∥ step 3 (GPU); model/dataset downloads background from step 1; step-6 hook design overlaps 5a stabilization. The 3 configs run sequentially (shared 32-core CPU).
