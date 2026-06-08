# B2 run log

Run record for the B2 / TierShift characterization benchmark (see [[Initial Plan 2]],
`code/b2/RUNBOOK.md`). Append per session.

## 2026-06-07 — first end-to-end run (capture + benches + sim + charts)

**Headline: B2 Gate 3 PASSED — expert load is strongly non-stationary and domain-sensitive.**
All capture-side go/no-go signals are favorable. The placement objective (Q2/Q4) is the only
piece still pending a real tier gap (see Tier bench below).

### Environment / cluster
- **watgpu308 (the canonical Ada+A6000 node) was draining for a reboot** ("Needs a reboot",
  set by root 2026-06-02) → no SCHOOL GPU jobs could land. Confirmed via `sinfo -n watgpu308 -R`.
- Pivoted to the **ALL partition** (any free GPU on WATGPU). Capture is single-GPU and
  hardware-independent (routing = model+input), so this is valid for Q1/Q5/Q6.
- Fleet visible via ALL (one GPU/node, owned by other groups): RTX 6000 Ada (108/608),
  L40S (408), **H200 NVL** (508), **RTX PRO 6000 Blackwell** (1008). Wider arch span than 308.
- Env built with **uv** (no conda on the cluster): `code/b2/.venv`, torch 2.11.0+cu130,
  vLLM fork @ `84969d1a4` (b2-expert-telemetry), OLMoE-1B-7B-0924-Instruct, BF16.
- **vLLM serve needed `VLLM_USE_FLASHINFER_SAMPLER=0` + `VLLM_ATTENTION_BACKEND=FLASH_ATTN`**:
  flashinfer JIT-compiles a sampler kernel at runtime and needs `nvcc`, which the uv env lacks
  (torch ships the CUDA *runtime*, not the toolkit). Disabling flashinfer's JIT paths fixed it.
  → folded into `code/b2/RUNBOOK.md` B3 and the capture job.

### B3 — expert-load capture (the headline)
- Trace: `make_trace` mixed drifting, 600 reqs (150 each chat/code/math/mixed), max 128 tok.
- Served OLMoE on 1× RTX 6000 Ada (watgpu108), hook on, `--enforce-eager`. Replay 8.9 req/s.
- `expert.jsonl`: **64,991 rows**, **67 windows**, 16 layers × 64 experts, top-8 — correct OLMoE.

| Metric | Value | Reading |
|---|---|---|
| Imbalance ratio swing | **2.12 → 8.00** | strongly non-stationary (Q1) |
| Per-segment mean imbalance | chat **3.08**, code **6.68**, math **5.51**, mixed **4.71** | hot set is domain-sensitive (Q1/Q6) |
| Reactive hit-rate (Q5) | **0.741** | last window predicts 74% of next window's hot set → reactive cache viable, no forecast |
| Temporal-mass fraction | **0.759** (25 consistent / 610 temporal) | ~76% of hot mass is bursty → large online-over-static headroom ceiling |
| Hot-set turnover (median) | **5.73 s** | the churn timescale |

### B5 — migration micro-bench (Q3 kill-test)
- Expert weight = **12.6 MB** (BF16). PCIe transfer **0.503 ms** (analytic Gen4 ~25 GB/s;
  real PCIe pending a 2-GPU node / 308).
- migration 0.5 ms ≪ turnover 5.73 s → **headroom ≈ 11,391× → MIGRATE VIABLE.** (Q3 pass.)

### B4 — tier micro-bench (the caveat)
- Cross-node single-GPU benches (new `tier_bench --single/--combine`): **H200 (fast) vs
  RTX 6000 Ada (slow)** → expert-FFN tok/s **ratio ≈ 1.0** at every batch (0.85× @b1 … 1.01× @b512).
- **Finding:** OLMoE's expert FFN (2048→1024) is small enough to be launch/overhead-bound on
  both modern cards (~134 TFLOPS achieved at b512 — neither H200 nor Ada saturates), so the
  tier gap ~vanishes *between modern datacenter GPUs*. The intended gap lives at the
  fast-vs-**Ampere A6000** boundary (and FP8-Ada vs BF16-Ampere), which needs **watgpu308**.
- Consequence: **Q2/Q4 (placement sim) are not meaningful at ratio 1.0** — with no tier gap,
  caching hot experts on "fast" buys nothing and reactive only pays migration cost (sim shows
  this faithfully: GEM gap −41%, make-or-break ~0, reactive worse than static). The pipeline is
  proven end-to-end; the real Q2/Q4 numbers await the Ada-vs-A6000 measurement on 308.
- Note: temporal-mass = 0.76 is a **workload** property (hardware-independent) — it says the
  headroom *exists*; whether reactive+split realizes it needs a real tier gap to land.

### Artifacts (on cluster, gitignored)
- Capture: `/u3/u3anand/b2/runs/capture/20260608T013125/`
- `tier_cost.json` (H200/Ada, ratio~1), `migration_cost.json`, `sim.json`, `charts/` (Q1–Q7 PNGs)
- All 7 charts render. Q1 (non-stationarity) and Q6 (domain heatmap) are clean and on-message.

### Follow-ups
- [ ] **Re-run B4/B5/B6 on watgpu308** (Ada fast + A6000 slow) when it's back → the real tier
  gap → meaningful Q2/Q4 + real PCIe migration ms. Sim/charts are parameterized; just re-point.
- [ ] Filter pre-first-segment **warmup windows** (vLLM startup profiling run pins imbalance at
  8.0 before the chat segment) for a cleaner Q1.
- [ ] Check trace composition (LMSYS is gated → some domains fall back to synthetic templated
  prompts). Drift is real (model routes them genuinely), but note real-vs-synthetic share.
- [ ] Optionally bench Blackwell (1008) / L40S (408, hit a CUDA-init error once) for a fuller
  arch sweep; measure real PCIe on a 2-GPU node (508 has several free).
