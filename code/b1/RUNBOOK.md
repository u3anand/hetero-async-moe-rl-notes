# RUNBOOK — B1 OLMoE benchmark run

Exact ordered steps to run the B1 initial benchmark (see `Initial Plan.md`). Two tracks:
**Track A** (fork edits) runs on the **laptop, no cluster needed** — do it while the node is busy.
**Track B** (cluster) runs on **watgpu308** during a free window. Track B5 (the instrumented
3-config runs) requires Track A to be merged first; B0–B4 (bring-up + spikes) do not.

## Pre-flight — is the node free?
```bash
ssh u3anand@watgpu.cs.uwaterloo.ca 'bash -s -- all' < code/b1/slurm/check-node.sh
```
Need **watgpu308 = READY** for the full run (PARTIAL is enough for B0–B4 spikes). As of 2026-05-31 it's BUSY → target the overnight window. Re-check before each session.

---

## Track A — fork edits (laptop, anytime)
Clone the forks locally (vLLM included — fine per decision), branch, edit, push. After each, advance the submodule pointer + `pins.txt`.
```bash
git -C "$REPO" submodule update --init code/forks/{prime-rl,vllm,mini-swe-agent}
git -C code/forks/<repo> checkout -b b1-instrumentation
# ...edit... ; git commit ; git push -u origin b1-instrumentation
git -C "$REPO" add code/forks/<repo> && git -C "$REPO" commit -m "bump <repo> pointer"
```
**A1 — mini-swe-agent:** add `UdockerEnvironment(Environment)` (model on `DockerEnvironment.execute()`; run `udocker run` via `code/b1/sandbox/run_test.sh`) + per-call timing/classification → `b1tel` episode stream. *No cluster needed; testable locally with udocker.*
**A2 — vLLM:** add a kernel-free hook at the MoE gate (`router_logits, _ = self.gate(...)` in the OLMoE/Qwen MoE block) to log per-token router logits + `layer_id`; gated by a sampling rate. Confirm exact model file once cloned.
**A3 — prime-rl:** emit `t_train` + broadcast router/expert split (trainer) and per-episode start/end + `policy_version`/`replica_id` + queue depth/staleness (orchestrator) → `b1tel`; wire the SWE env's tool exec through `UdockerEnvironment`; verify per-replica `CUDA_VISIBLE_DEVICES` pinning.

**Without A:** B2/B3/B4 still run on stock forks (sandbox timings = Q1/Q5 anchors; inference fast:slow ratio; train C_T baseline). **Q4/Q6 and the full instrumented loop need A merged.**

---

## Track B — cluster session (watgpu308, overnight)

**B0 — recon** (`salloc -p SCHOOL -w watgpu308 --gres=gpu:1 -t 0:30:00`): `nvidia-smi -L` → record index→tier map; confirm `eth0`; `df -h /tmp` (node scratch). **Gate 0:** map = {4 L40S, 2 A6000, 2 RTX6000 Ada}; eth0 confirmed.

**B1 — env** (`salloc -p SCHOOL -w watgpu308 --gres=gpu:2 --cpus-per-task=16 --mem=128G -t 7:00:00`):
```bash
bash code/b1/setup.sh
```
**Gate 1:** torch/vLLM/flash-attn import on a compute node; `udocker run alpine` ok; 2-GPU NCCL all-reduce completes with `NCCL_SOCKET_IFNAME=eth0` (no hang).

**B2 — sandbox spike** (CPU, can run ∥ with B3):
```bash
python code/b1/sandbox/make_images_list.py --n 15 --out code/b1/sandbox/images.txt   # see note
bash code/b1/sandbox/prewarm.sh code/b1/sandbox/images.txt
# then run each container's pytest, record pull_s/setup_s/test_wall_s/cores/exit_code
```
**Gate 2:** ≥80% run to a clean pass/fail under udocker R2; **cores-per-test recorded** (sets the rollout concurrency cap). Fallback: exec mode P2; stage `UDOCKER_DIR` to node-local scratch if NFS extraction is slow. → first **Q1/Q5** anchors.

**B3 — inference spike:** `vllm serve allenai/OLMoE-1B-7B-... --dtype bfloat16` on 1 Ada (BF16, no TP); drive a ~100-turn long-context prompt → KV fit + ~1.5 s/turn; then serve on 1 A6000 → **record fast:slow tokens/s ratio**; probe router-logit extraction (informs A2). **Gate 3.**

**B4 — train spike** (`D_T` = 2 L40S): one prime-rl FSDP2+LoRA GRPO step on OLMoE; write a checkpoint; kill + resume. Record C_train baseline. **Gate 4.**

**B5 — integration + 3-config runs** *(requires Track A merged)*:
```bash
for C in homo-fast homo-slow hetero; do
  sbatch --export=ALL,CONFIG=$C code/b1/slurm/run-config.sbatch
done
```
Run each to a ~30–60 min steady state; emit the three telemetry streams; survive a simulated preempt (`scancel --signal=SIGUSR1 <jobid>` → requeue → resume). Then:
```bash
python -m b1tel.charts --runs homo-fast=$B1/runs/homo-fast/<ts> \
  homo-slow=$B1/runs/homo-slow/<ts> hetero=$B1/runs/hetero/<ts> --out code/b1/charts
```
**Gate 5/6:** all three configs reach steady state; all Q1–Q7 charts render.

---

## Tonight's realistic scope
With Track A **not yet merged**: do **B0–B4** (full bring-up + sandbox/inference/train spikes) on the free overnight node — this de-risks the entire stack on real hardware and yields the first anchors (Q1/Q5 from sandbox, fast:slow ratio from inference, C_T from train). **B5 (the instrumented 3-config benchmark) is the next session, after Track A lands.** If Track A is finished on the laptop before the window, B5 can run the same night.

## Record
Append outcomes (gate pass/fail, anchor numbers, the fast:slow ratio, cores-per-test) to a new `B1 run log.md` note in the vault; commit final charts to `code/b1/charts`; update `pins.txt` with the tested commits + udocker version.

## Manifests checklist (ready before a run)
- [x] `setup.sh` (idempotent bootstrap) · `slurm/run-config.sbatch` + `_env.sh` · `slurm/check-node.sh`
- [x] `configs/{train,orch,infer-fast,infer-slow,infer-hetero}.toml` — **TEMPLATES**: reconcile field names against `code/forks/prime-rl/examples/*` during B1
- [x] `sandbox/{prewarm,run_test}.sh` + `make_images_list.py`; fill `images.txt` in B2
- [x] `pins.txt` — prime-rl/mini-swe-agent pinned; **re-pin vLLM to prime-rl's version** during B1
- [ ] Track A fork edits merged (gates B5)
