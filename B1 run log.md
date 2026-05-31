# B1 run log

Outcomes from B1 cluster sessions. Append per session. See [[Initial Plan]] / `code/b1/RUNBOOK.md`.

---

## 2026-05-31 — Tier-1 smoke: telemetry + CUDA trace on watgpu308 ✅

**Goal:** first real-hardware signal — confirm our B1 step-stream telemetry + the torch CUDA
trace come out, on the small model. Trainer-only, fake data, 1 GPU (L40S). No orchestrator /
inference / udocker / SWE (those are Tier 2).

**Result: SMOKE OK ✅** — 5 fake-data steps, exit 0, peak 22 GiB. All three checks pass:
- `step.jsonl` — 5 rows w/ `t_train` (Q3). e.g. `{"step_id":0,"t_train":3.15,"broadcast":null,"mfu":0.307,"tokens":2048,...}`
- `system.jsonl` — 34 per-GPU rows (sampler): `{"gpu_index":0,"sm_active":..,"dram_active":..,"mem_used_mib":..,"cpu_util":..}`
- `trace_0.json.gz` — 6.6 MB torch CUDA timeline, opens cleanly.
- `broadcast` is `null` on every row — **expected** trainer-only (no weight broadcast without inference); Q4 split is Tier 2.

**Artifacts:** `/u3/u3anand/b1/runs/smoke/20260531T215542/` (telemetry/ + trace/). Harness:
`code/b1/{setup-node.sh, configs/smoke-train.toml, slurm/smoke-trainer.sh}`. Re-run (env built):
`srun -p SCHOOL -w watgpu308 --gres=gpu:1 -t 0:30:00 --cpus-per-task=8 --mem=64G bash code/b1/slurm/smoke-trainer.sh`.

### Findings (the smoke surfaced these — several matter beyond Tier 1)
1. **Cluster was bare** (no conda/uv/repo). Bring-up is uv-based (`setup-node.sh`), Python 3.12,
   ~5 min incl. OLMoE download. Conda `setup.sh` is superseded.
2. **No GitHub key on the node**; repos are public → `git config --global url."https://github.com/".insteadOf "git@github.com:"` lets the SSH-form submodules clone over HTTPS.
3. **flash-attn is required even with `attn="sdpa"`** — prime-rl's model registry imports
   `ring_flash_attn`→`flash_attn` at module load. Installed via the `flash-attn` extra (prebuilt
   cu128/torch2.11/cp312 wheel — no build).
4. **⚠ OLMoE pretrained weights don't load in prime-rl's trainer.** The HF checkpoint stores
   per-expert weights but prime-rl's MoE expects a fused `experts.down_proj`; OLMoE has **no
   custom trainer impl** in this fork → `Missing key ... experts.down_proj`. **Blocks B4/B5 real
   OLMoE training** until resolved (weight converter, or add a custom OLMoE impl). Smoke used
   `model.debug.random_init=true` to bypass.
5. **⚠ Full 7B OLMoE OOMs one 44 GB L40S** — trainer upcasts to fp32 (`optimization_dtype`, not
   ours to change): params+AdamW ≫ 44 GB. Smoke truncated to `num_layers=2`. For B4: needs
   **LoRA** (the plan's intent) and/or multi-GPU FSDP — full-param single-GPU 7B is infeasible.
6. **`uv run`/`uv sync` drop our editable `b1tel`** (not in prime-rl's lock) → silent no-op
   telemetry. Launcher now uses `.venv/bin/{python,torchrun}` directly + reinstalls b1tel.
7. **Kineto/CUPTI profiler segfault was an OOM symptom**, not an env bug — vanished once the
   model fit. `trace_path` works; default `TRACE=1`, set `TRACE=0` to skip the profiler.

### Validated vs not
- **Validated (Tier 1):** uv env on watgpu308, OLMoE arch fwd/bwd on L40S, trainer step telemetry
  (Q3 `t_train`/`mfu`/`tokens`), sampler (`system.jsonl`), torch CUDA trace, b1tel↔prime-rl seam.
- **Not yet:** Q4 broadcast split (needs inference/weight broadcast), orchestrator episode/queue
  (Q1/Q2/Q5/Q6, needs udocker+SWE), real OLMoE weights (finding #4), LoRA/multi-GPU (finding #5),
  vLLM A2 hook (placeholder pin).

### Next
- Decide OLMoE trainer support (converter vs custom impl) — gates B4/B5.
- Tier 2: 2-GPU `uv run rl` + udocker SWE → exercises Q1/Q2/Q4/Q5/Q6 + tool-call timing.
- Optional: NVTX ranges (fwd/bwd/opt/broadcast + A2 gate) for a labeled nsys timeline.
