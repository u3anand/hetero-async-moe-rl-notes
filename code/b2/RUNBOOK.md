# RUNBOOK — B2 MoE expert-hotness × tier profiling

Exact ordered steps to run the B2 initial benchmark (see vault `Initial Plan 2.md`). Two tracks:
**Track A** (the one vLLM fork edit) runs on the **laptop, no cluster** — and is **shared with
B1-A2**, so it's not wasted whichever direction wins. **Track B** (cluster) runs on **watgpu308**.
Unlike B1, B2 has **no trainer / no rollout agent / no udocker** — it's serve + capture + offline math.

## Pre-flight — is the node free?
```bash
ssh u3anand@watgpu.cs.uwaterloo.ca 'bash -s -- all' < code/b2/slurm/check-node.sh
```
B2 bring-up needs only **1 Ada + 1 A6000** → even **PARTIAL/LIMITED** (≥1 free GPU on 308) is enough
for B0–B5; **READY** lets you also do the live B7 validation. Re-check before each session.
**Status 2026-06-07 15:27Z: watgpu308 = READY (8/8 idle).**

---

## Track A — the gate hook (laptop, anytime; shared with B1-A2)
Add a kernel-free hook at the MoE gate to log per-expert token load. Same insertion point B1 needs.
```bash
git -C "$REPO" submodule update --init code/forks/vllm
git -C code/forks/vllm checkout -b b2-expert-telemetry      # or extend b1-instrumentation
```
**A1 — vLLM MoE-gate hook:** in the OLMoE / Qwen3-MoE block, right after `router_logits, _ =
self.gate(hidden)` and the top-k select, increment a per-`(layer_id, expert_id)` token counter into a
ring buffer; a sampler thread flushes `(window_id, layer_id, expert_id, token_load, segment_label)`
to `b2tel`'s `expert.jsonl` at a fixed window cadence. Gated by a sampling rate; **no kernel
changes** (`VLLM_USE_PRECOMPILED=1`). Confirm the exact model file once cloned. Push; bump pointer +
`pins.txt`.

**Without A:** B4/B5 (the micro-benches) still run on stock vLLM; only the Q1/Q5/Q6 capture (B3) needs A.

---

## Track B — cluster session (watgpu308)

**B0 — recon** (`salloc -p SCHOOL -w watgpu308 --gres=gpu:1 -t 0:30:00`): `nvidia-smi -L` → record
index→tier map; pick one **Ada** (fast) and one **A6000** (slow) CUDA index. **Gate 0:** map =
{4 L40S, 2 A6000, 2 RTX6000 Ada}; fast+slow indices chosen.

**B1 — env** (`salloc -p SCHOOL -w watgpu308 --gres=gpu:2 --cpus-per-task=8 --mem=64G -t 7:00:00`):
```bash
bash code/b2/setup.sh        # conda env + vLLM fork + OLMoE download (no udocker, no prime-rl)
```
**Gate 1:** vLLM imports on a compute node; `vllm serve allenai/OLMoE-1B-7B-0924-Instruct
--dtype bfloat16` answers one request on the Ada index.

**B2 — trace build** (CPU, ∥ B1):
```bash
python code/b2/src/b2tel/make_trace.py --chat sharegpt,lmsys --code humaneval --math gsm8k \
  --segments chat,code,math,mixed --out /u3/u3anand/b2/traces/mixed.jsonl
```
**Gate 2:** manifest replays; per-request `segment_label` present (the Q6 domain tags).

**B3 — expert-load capture** (GPU, 1 Ada) — *requires Track A*:
```bash
vllm serve allenai/OLMoE-1B-7B-0924-Instruct --dtype bfloat16 \
  --port 8000 &   # fork build, gate hook on, B2TEL_OUT=/u3/u3anand/b2/runs/capture/<ts>
python code/b2/src/b2tel/replay.py --trace /u3/u3anand/b2/traces/mixed.jsonl --url :8000
```
**Gate 3 (headline):** `expert.jsonl` populated; **imbalance ratio visibly shifts across segments.**
If flat → expert load is stationary → B2 = negative result, reconsider direction.

**B4 — tier micro-bench** (1 Ada + 1 A6000):
```bash
python code/b2/src/b2tel/tier_bench.py --fast <ada_idx> --slow <a6000_idx> \
  --batch 1,8,32,128 --out /u3/u3anand/b2/runs/tier_cost.json
```
Times one expert FFN (and a small all-to-all) per tier. **Gate 4:** fast:slow tokens/s ratio recorded
per batch size (expect ≥2.4×; larger if FP8-Ada vs BF16-Ampere).

**B5 — migration micro-bench** (1 Ada + 1 A6000) — *the kill-test*:
```bash
python code/b2/src/b2tel/migrate_bench.py --fast <ada_idx> --slow <a6000_idx> \
  --capture /u3/u3anand/b2/runs/capture/<ts>/expert.jsonl \
  --out /u3/u3anand/b2/runs/migration_cost.json
```
Measures expert-weight bytes + PCIe Ada↔A6000 transfer ms; derives hot-set turnover period from the
capture. **Gate 5 (Q3):** `migration_ms` vs `turnover_period` recorded → **migrate viable iff
migration ≪ turnover; else replication-only.** This decision shapes the controller.

**B6 — offline simulate + charts** (CPU):
```bash
python -m b2tel.simulate --capture .../expert.jsonl --tier-cost .../tier_cost.json \
  --migration .../migration_cost.json --fleet-cost code/b2/fleet_cost.json \
  --slo-p99-ms <target> \
  --configs static-balanced,static-hot-on-fast,reactive-LRU-port,reactive+split,oracle-dynamic
python -m b2tel.charts --runs /u3/u3anand/b2/runs --out code/b2/charts
```
Each config is scored on the **objective**: **p99 latency / SLO-attainment** and **$/token** (from
`fleet_cost.json`), not just straggler. **Gate 6:** under drift, static violates the p99 SLO while
reactive holds it; **$/token-at-SLO beats the all-fast-GPU reference**; the **reactive-LRU-port →
reactive+split** headroom (make-or-break) is non-trivial; Q1–Q7 charts render.

**B7 — live EP validation** *(optional; 1 Ada + 1 A6000)*: serve OLMoE EP across both tiers under
`static-balanced`; sample `system.jsonl`; confirm measured per-tier straggler ≈ simulated. **Gate 7:**
within tolerance → offline model trusted for controller work.

---

## Tonight's realistic scope
With Track A **merged**: **B0→B6** (capture + both micro-benches + offline charts) in one session —
that's the **entire headline result** (Q1/Q2/Q3). B7 (live validation) optional same night if 308
stays held. Without Track A: do **B0,B1,B2,B4,B5** (env + both micro-benches + trace) — everything
except the Q1 capture — then write A on the laptop and run B3/B6 next session.

## Record
Append outcomes (gate pass/fail, imbalance-ratio swing, fast:slow ratio, migration-ms,
turnover-period, static↔oracle gap) to a new `B2 run log.md` note in the vault; commit final charts to
`code/b2/charts`; update `pins.txt` with the tested vLLM commit.

## Manifests checklist (build before a run)
- [x] `slurm/check-node.sh` (reused from B1, cluster-wide)
- [ ] `setup.sh` — conda env + vLLM fork + OLMoE download (simpler than B1's: no udocker/prime-rl)
- [ ] `src/b2tel/{telemetry,make_trace,replay,tier_bench,migrate_bench,simulate,charts}.py`
- [ ] Track A vLLM gate hook merged (gates B3 only; micro-benches don't need it)
- [ ] `pins.txt` — pin the vLLM commit
