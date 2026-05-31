# b1 — instrumentation for the OLMoE bring-up

Code for the initial benchmark (see vault note `Initial Plan.md`): bring the async MoE-RL loop
up on watgpu308 with OLMoE-1B-7B under three configs (homo-fast / homo-slow / hetero) and emit
the Q1–Q7 telemetry.

## What's here (our own code)
- `src/b1tel/telemetry.py` — JSONL writers + schemas for the three streams (episode / step / system). The contract everything joins on.
- `src/b1tel/sampler.py` — ~1 Hz system sidecar (GPU sm%/HBM%, sandbox queue depth, CPU).
- `src/b1tel/charts.py` — Q1–Q7 figures from the JSONL (3-config side-by-side).
- `configs/` — prime-rl TOMLs: `train.toml`, `orch.toml`, `infer-{fast,slow,hetero}.toml` (differ only in rollout GPU pinning).
- `slurm/` — sbatch templates (`_env.sh` shared header + per-step scripts).
- `sandbox/` — udocker pull/prewarm + run-test helper used by the agent's reward.
- `setup.sh` — WatGPU bootstrap (conda env, fork submodules, udocker, downloads).
- `pins.txt` — lockfile: exact commits of the three forks + udocker.

## Forks (owned, as submodules under `../forks/`)
`u3anand/{prime-rl, vllm, mini-swe-agent}`. Fork edits:
- **prime-rl**: C_T + broadcast-split + queue/staleness emits; route the SWE env's tool exec through udocker.
- **vllm**: one kernel-free Python hook at the MoE gate to log router logits (Q6); install `VLLM_USE_PRECOMPILED=1`.
- **mini-swe-agent**: `UdockerEnvironment` + per-tool-call timing/classification.

## Run artifacts
Models, dataset, runs, telemetry JSONL, and `.udocker` live on WatGPU at `/u3/u3anand/b1` (gitignored).
Only final selected charts + a results note are committed back to the vault.
