#!/usr/bin/env bash
# WatGPU bootstrap for the B1 OLMoE bring-up. Run INSIDE an interactive job
# (login node is noexec on NFS + kills heavy procs):
#   salloc -p SCHOOL --gres=gpu:2 -t 7:00:00 --cpus-per-task=16 --mem=128G
#   bash code/b1/setup.sh
# Idempotent: safe to re-run.
set -euo pipefail

B1=${B1_ROOT:-/u3/u3anand/b1}
REPO=${REPO_ROOT:-$HOME/research/notes/Research\ v0}      # this notes repo on the cluster
ENV=${CONDA_ENV:-b1}
export HF_HOME=$B1/models
export UDOCKER_DIR=$B1/sandbox/.udocker
export NCCL_SOCKET_IFNAME=eth0

mkdir -p "$B1"/{models,data,sandbox,runs}

echo "== conda env =="
if ! conda env list | grep -q "^$ENV "; then
  conda create -y -n "$ENV" python=3.11
fi
# shellcheck disable=SC1091
source activate "$ENV"
conda install -y -c nvidia cuda=12.4 || true   # provides nvcc; harmless if already present

echo "== fork submodules (pinned in pins.txt) =="
git -C "$REPO" submodule update --init --recursive code/forks/prime-rl code/forks/vllm code/forks/mini-swe-agent

echo "== vLLM fork (kernel-free edit → skip CUDA build) =="
VLLM_USE_PRECOMPILED=1 pip install -e "$REPO/code/forks/vllm"

echo "== prime-rl fork (pointed at our vLLM) + mini-swe-agent fork =="
pip install -e "$REPO/code/forks/prime-rl"
pip install -e "$REPO/code/forks/mini-swe-agent"

echo "== our telemetry package =="
pip install -e "$REPO/code/b1"

echo "== udocker (rootless sandbox) =="
pip install --user udocker
python -m pip show udocker >/dev/null 2>&1 || true
"$HOME/.local/bin/udocker" install || python -c "import udocker" 2>/dev/null || true

echo "== downloads (background-friendly) =="
huggingface-cli download allenai/OLMoE-1B-7B-0924-Instruct --local-dir "$HF_HOME/OLMoE-1B-7B" || true
huggingface-cli download nebius/SWE-rebench-openhands-trajectories --repo-type dataset --local-dir "$B1/data/swe-rebench" || true

echo "== sanity =="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
import b1tel; print("b1tel ok")
PY
echo "setup.sh done. Pins:"; cat "$REPO/code/b1/pins.txt"
