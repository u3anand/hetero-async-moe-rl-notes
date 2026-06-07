#!/usr/bin/env bash
# WatGPU bootstrap for the B2 OLMoE expert-hotness × tier benchmark. Run INSIDE an
# interactive job (login node is noexec on NFS + kills heavy procs):
#   salloc -p SCHOOL -w watgpu308 --gres=gpu:2 --cpus-per-task=8 --mem=64G -t 7:00:00
#   bash code/b2/setup.sh
# Simpler than B1: serving-only — no udocker, no prime-rl, no mini-swe-agent.
# Idempotent: safe to re-run.
set -euo pipefail

B2=${B2_ROOT:-/u3/u3anand/b2}
REPO=${REPO_ROOT:-$HOME/research/notes/Research\ v0}      # this notes repo on the cluster
ENV=${CONDA_ENV:-b2}
export HF_HOME=$B2/models
export NCCL_SOCKET_IFNAME=eth0                            # only matters for multi-GPU B7

mkdir -p "$B2"/{models,data,traces,runs}

echo "== conda env =="
if ! conda env list | grep -q "^$ENV "; then
  conda create -y -n "$ENV" python=3.11
fi
# shellcheck disable=SC1091
source activate "$ENV"

echo "== vLLM fork (kernel-free MoE-gate hook → skip CUDA build) =="
git -C "$REPO" submodule update --init code/forks/vllm
VLLM_USE_PRECOMPILED=1 pip install -e "$REPO/code/forks/vllm"

echo "== our telemetry/analysis package (numpy + matplotlib + orjson) =="
pip install -e "$REPO/code/b2"
pip install "datasets>=2.18" || true   # for make_trace.py (optional; synthetic fallback exists)

echo "== OLMoE download (BF16; the bring-up model) =="
huggingface-cli download allenai/OLMoE-1B-7B-0924-Instruct \
  --local-dir "$HF_HOME/OLMoE-1B-7B" || true

echo "== sanity =="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
import b2tel
print("b2tel ok:", b2tel.EXPERT_FIELDS[:6], "...")
PY

echo
echo "setup.sh done. Next (see code/b2/RUNBOOK.md):"
echo "  B3 capture: VLLM_B2_EXPERT_LOG_DIR=$B2/runs/capture/<ts> \\"
echo "    vllm serve allenai/OLMoE-1B-7B-0924-Instruct --dtype bfloat16 --enforce-eager --port 8000"
echo "  then: python -m b2tel.replay --trace $B2/traces/mixed.jsonl --url :8000 --out $B2/runs/capture/<ts>"
echo "Pins:"; cat "$REPO/code/b2/pins.txt"
