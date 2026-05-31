#!/usr/bin/env bash
# B1 Tier-1 node bring-up (uv, no conda). Run INSIDE an salloc compute job — the login
# node is noexec on NFS and kills heavy procs:
#   salloc -p SCHOOL -w watgpu308 --gres=gpu:1 -t 2:00:00 --cpus-per-task=8 --mem=64G
#   bash code/b1/setup-node.sh
# Idempotent. Minimal scope: trainer-only fake-data smoke (telemetry + CUDA trace). Skips
# vLLM, udocker, and the SWE dataset (those are Tier 2). Supersedes the conda setup.sh.
set -euo pipefail

# Repo root = two levels up from this script (code/b1/ -> repo root).
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
FORK="$REPO/code/forks/prime-rl"
B1=${B1_ROOT:-/u3/u3anand/b1}
export HF_HOME=${HF_HOME:-$B1/models}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$B1"/{models,runs}

echo "== [1/6] uv =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1090
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
command -v uv >/dev/null || { echo "uv not on PATH after install"; exit 1; }
uv --version

echo "== [2/6] prime-rl fork submodule (notes-repo level) =="
git -C "$REPO" submodule update --init code/forks/prime-rl

echo "== [3/6] prime-rl's own deps submodules (NOT --recursive: avoids private configs/private) =="
git -C "$FORK" submodule update --init -- \
  deps/verifiers deps/renderers deps/research-environments deps/pydantic-config

echo "== [4/6] venv (py3.12) + uv sync (+flash-attn extra) =="
cd "$FORK"
uv venv --python 3.12
# flash-attn is a PREBUILT wheel (cu128/torch2.11/cp312 → matches our torch); no CUDA build.
# Required even though we run attn=sdpa: prime_rl's model registry imports ring_flash_attn ->
# flash_attn unconditionally at module load. --extra flash-attn keeps the install minimal
# (no --all-extras → skips the verifiers envs we don't need for the trainer-only smoke).
uv sync --extra flash-attn

echo "== [5/6] b1tel telemetry package (editable) =="
uv pip install -e "$REPO/code/b1"

echo "== [6/6] OLMoE-1B-7B download (model only; skip SWE dataset) =="
uv run python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download("allenai/OLMoE-1B-7B-0924-Instruct")
print("OLMoE present under", os.environ.get("HF_HOME"))
PY

echo "== sanity =="
uv run python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
import b1tel
from b1tel.telemetry import TelemetryLogger  # noqa: F401
print("b1tel import OK")
PY
echo "setup-node.sh done. Next: bash $REPO/code/b1/slurm/smoke-trainer.sh"
