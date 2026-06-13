# Qwen3-30B-A3B capture — run conditions (WatGPU)

The exact environment for the B2 Qwen capture. Recorded because the WatGPU stack
(torch 2.11 / cu130, **no `nvcc`**, venv originally built for OLMoE) forced several
**kernel-backend** overrides to get vLLM to serve Qwen3-30B-A3B at all. **None of these touch
the MoE router** — they only choose which kernel computes attention / expert-FFN / sampling —
so the routing trace (`expert.jsonl`) is faithful. They are routing-irrelevant by construction.

## Model / serving
- Model: `Qwen/Qwen3-30B-A3B`, **BF16**, single **H200 NVL** (watgpu708), `tp=1`.
- `--enforce-eager` (the telemetry hook no-ops under CUDA graph / torch.compile).
- `--max-model-len 16384`, default `gpu-memory-utilization` (60 GB weights on 143 GB card).
- 48 layers, 128 experts, top-8 (confirmed by the hook-validity gate).

## venv dependency added (one-time)
- `flashinfer-python==0.6.11.post2` (pairs with the preexisting `flashinfer-cubin` — AOT
  prebuilt cubins, so **no nvcc** at runtime). Installed with
  `uv pip install --python code/b2/.venv/bin/python flashinfer-python==0.6.11.post2`.
  Needed because vLLM V1 imports the FlashInfer backend module during engine init.

## Required env / flags on WatGPU (no nvcc → avoid all JIT-compiling kernels)
```
export VLLM_ATTENTION_BACKEND=TRITON_ATTN     # flashinfer attn would JIT-compile (nvcc)
export VLLM_USE_FLASHINFER_SAMPLER=0          # flashinfer sampler would JIT-compile
export VLLM_USE_DEEP_GEMM=0                    # deep_gemm FP8 kernels absent
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip             # warmup probe raises if deep_gemm missing
vllm serve Qwen/Qwen3-30B-A3B --dtype bfloat16 --enforce-eager \
     --moe-backend triton --max-model-len 16384   # flashinfer cutlass MoE would JIT (nvcc)
```
Why each: vLLM "auto" selects flashinfer/cutlass/deep_gemm kernels on Hopper when the libs are
importable; those paths JIT-compile with nvcc, which WatGPU lacks. Forcing the **Triton**
kernels (triton is installed) sidesteps every JIT. The MoE gate (`router_logits =
gate(hidden_states)` + top-8) runs in our model code regardless of these, so routing is unchanged.

## Telemetry env (the B2 hook)
```
export VLLM_B2_EXPERT_LOG_DIR=<capture-dir>   # enables expert.jsonl + expert_mb.jsonl
export VLLM_B2_WINDOW_S=1.0                    # window length (s)
export VLLM_B2_SIM_GPUS=8                      # round-robin EP layout D for per-GPU imbalance
```
Prefill/decode phase is derived per-token from `query_start_loc` (vLLM V1 has no
`num_prefill_tokens`); qlen==1 ⇒ decode.

## Workload
```
python -m b2tel.make_trace --chat sharegpt --code humaneval --math gsm8k \
    --segments chat,code,math,mixed --per-segment 200 --length-mix short,medium,long --seed 0 \
    --out traces/qwen-mixed.jsonl                       # 800 reqs, real datasets, varied lengths
python -m b2tel.replay --trace traces/qwen-mixed.jsonl --model Qwen/Qwen3-30B-A3B \
    --rate 20 --concurrency 64 --out <capture-dir>      # representative load level
```

## Pinned scripts on the cluster
`/u3/u3anand/b2/runs/qwen/{capture.sbatch, gate_check.py, verdict.py}` — capture + gates +
the make-or-break verdict numbers.
