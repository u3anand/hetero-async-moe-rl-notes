# b2 — instrumentation for MoE expert-hotness × tier profiling

Code for the B2 initial benchmark (see vault note `Initial Plan 2.md`): serve a fixed MoE on
watgpu308, replay a mixed drifting request trace, capture per-expert load, micro-bench the tier gap
and migration cost, and simulate static-vs-dynamic placement to emit Q1–Q7 telemetry.

**B2 vs B1:** serving-only. **No prime-rl, no mini-swe-agent, no udocker** — just vLLM (EP) + trace
replay + offline simulation. The first headline result is **single-GPU capture + two micro-benches +
offline math**; live cross-tier EP serving is the last (optional) validation gate, not a prerequisite.

## What's here (our own code)
- `src/b2tel/telemetry.py` — JSONL writer + schema for `expert.jsonl` (and `system.jsonl` in live B7).
- `src/b2tel/make_trace.py` — assemble the mixed ShareGPT/LMSYS + code + math trace into drifting segments.
- `src/b2tel/replay.py` — replay the trace against a vLLM endpoint at a controlled rate.
- `src/b2tel/tier_bench.py` — time one expert FFN (+ small all-to-all) on Ada vs A6000 → `tier_cost.json`.
- `src/b2tel/migrate_bench.py` — expert-weight bytes + PCIe Ada↔A6000 transfer ms; hot-set turnover → `migration_cost.json`.
- `src/b2tel/simulate.py` — join load × tier-cost × migration-cost × fleet-cost; score placement configs on **p99 latency / SLO-attainment + $/token** (the objective), not just straggler.
- `fleet_cost.json` — $/hr per tier (L40S vs A6000 list price) for the $/token / cost-at-SLO accounting.
- `src/b2tel/charts.py` — Q1–Q7 figures.
- `slurm/check-node.sh` — cluster availability (reused from b1, identical).
- `setup.sh` — WatGPU bootstrap (conda env, vLLM fork, OLMoE download). *(to write)*
- `pins.txt` — the tested vLLM commit. *(to write)*

## Fork (owned, shared with b1)
`u3anand/vllm` — the **MoE-gate hook** logging per-expert token load is the *same* hook B1-A2 needs;
B2 extends it to a windowed per-expert counter → `expert.jsonl`. No other fork required.

## Run artifacts
Models, traces, runs, telemetry JSONL live on WatGPU at `/u3/u3anand/b2` (gitignored). Only final
selected charts + a results note are committed back to the vault.
