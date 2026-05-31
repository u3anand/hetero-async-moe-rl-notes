# TRACK A — fork instrumentation edits (self-contained handoff)

Goal: make the three forks emit the B1 telemetry, entirely from the **laptop** (no cluster).
Write + push from here; **validation is on the cluster** (macOS can't run udocker / vLLM / GPU —
see test boundaries). When all three land, Track B5 (the instrumented 3-config run) is unblocked.

Repo root (this notes repo): `~/research/notes/Research v0`. Forks are submodules under
`code/forks/{prime-rl,vllm,mini-swe-agent}` → `u3anand/*`. Telemetry contract:
`code/b1/src/b1tel/telemetry.py` (`TelemetryLogger.episode/step/system`, `classify_command`,
`EPISODE_FIELDS/STEP_FIELDS/SYSTEM_FIELDS`, `TOOL_CATEGORIES`). Schema is also in `Initial Plan.md` → Instrumentation.

## Setup (once)
```bash
cd ~/research/notes/"Research v0"
git submodule update --init code/forks/mini-swe-agent code/forks/prime-rl code/forks/vllm  # vLLM is big
for r in mini-swe-agent prime-rl vllm; do
  git -C code/forks/$r checkout -b b1-instrumentation
done
```
Per edit, when done: `git -C code/forks/<r> commit -am ...; git -C code/forks/<r> push -u origin b1-instrumentation`,
then bump the pointer: `git add code/forks/<r> && git commit -m "bump <r>"`, and update `code/b1/pins.txt`.

Do them in this order (simplest → most source-reading):

---

## A1 — mini-swe-agent: UdockerEnvironment + tool timing (Q1)
**Seam (confirmed):** `src/minisweagent/__init__.py` defines `Environment` Protocol:
`execute(self, action: dict, cwd: str = "") -> dict[str, Any]` returning `{output, returncode, ...}`.
Implementations: `src/minisweagent/environments/local.py` (`LocalEnvironment`, subprocess) and
`docker.py` (`DockerEnvironment`, runs `docker exec -w cwd <container> bash -lc <command>`). The
single chokepoint for every command is `execute()` (and `DefaultAgent.execute_actions()` in
`agents/default.py`).

**Edit:** add `src/minisweagent/environments/udocker.py` → `UdockerEnvironment` modeled on
`DockerEnvironment`, but exec via `udocker run -w <cwd> <container> bash -lc <command>` (or shell
out to `code/b1/sandbox/run_test.sh`). In `execute()`, wrap with `time.perf_counter()` and
`b1tel.telemetry.classify_command(cmd)` → append a `tool_call` record `{category, t_start, t_end,
dur_s, cores, in_udocker=True}` to the active episode (Q1). Register it in the env factory/config
so prime-rl can select `executor="udocker"`.

**Test (local, macOS-OK):** unit-test command construction + `classify_command` mapping; do NOT
need a real container. Real udocker exec is validated on the cluster in Track B2.

## A2 — vLLM: router-logit logging hook (Q6)
**Seam (confirmed):** the MoE gate computes `router_logits, _ = self.gate(hidden_states)` then calls
the fused MoE. For **OLMoE** the file is `vllm/model_executor/models/olmoe.py` (the
`Olmoe...SparseMoeBlock.forward`); the Qwen analogue is `qwen2_moe.py` (`Qwen2MoeSparseMoeBlock`).
No public API exposes per-token logits (`--enable-return-routed-experts` gives only expert IDs +
has perf issues) → **patch the forward**.

**Edit:** right after the gate, if a logging flag is on (env var or model config), copy
`router_logits` (detached, to CPU async) with `layer_id` + a subset-sample rate, and append to a
per-process JSONL (or hand to `b1tel`). Keep it **kernel-free** (pure Python; preserves
`VLLM_USE_PRECOMPILED=1`). The orchestrator stamps `replica_id` + `router_version`.

**Test (local): write + push only.** Building/running vLLM needs Linux+CUDA — validated on the
cluster in Track B3 (the inference spike already probes router-logit extraction).
**Pin:** after editing, re-pin the fork to the vLLM version prime-rl uses (check
`code/forks/prime-rl` deps on the cluster), then rebase this one-file change onto it.

## A3 — prime-rl: C_T / broadcast-split / queue+staleness emits + udocker wiring (Q3/Q4/Q5/Q6 tags)
**Seams (inferred — verify in the cloned source):** trainer loop `src/prime_rl/trainer/rl/train.py`
(time fwd+bwd+opt → `step.t_train`; the weight-sync/`update_weights` path → split tensors
router-vs-expert by module name, time each NCCL group → `step.broadcast`). Orchestrator
`src/prime_rl/orchestrator/orchestrator.py` (rollout dispatch → per-episode `t_start/t_end` +
`policy_version` + `replica_id`; the rollout buffer/queue → `sandbox_queue_depth`, `batch_staleness`).
Config is **TOML+pydantic**; entrypoints `prime_rl.entrypoints.{trainer,orchestrator,inference,rl}`.
Wire the SWE/verifiers env's tool exec → `UdockerEnvironment` from A1 (the `mini-swe-agent-plus` env
in the Environments Hub is the integration point).

**Edit:** add the emit calls (piggyback the existing W&B logger surface) reading the `[telemetry]`
config keys already in `code/b1/configs/{train,orch}.toml`; emit via `b1tel.TelemetryLogger`.

**Test (local):** import + unit-test the emit/split helpers on dummy tensors. Full async run is the
cluster (Track B4/B5).

---

## Track A done when
All three forks have a pushed `b1-instrumentation` branch, the notes-repo submodule pointers + `pins.txt`
are bumped to those commits, and the local unit tests (command classification, broadcast-split,
router-logit buffering) pass. Then Track B5 can run the instrumented 3-config benchmark.

## Honest boundary
Only logic that runs without GPU/Linux is testable on the macOS laptop. udocker exec, vLLM serving,
and the prime-rl async loop are validated on watgpu308 (Track B). So Track A = **write + push + unit-test**;
real validation is the cluster bring-up.
