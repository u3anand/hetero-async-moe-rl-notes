#!/usr/bin/env bash
# WatGPU bootstrap for the B2 OLMoE expert-hotness × tier benchmark (uv, no conda — matches
# the cluster, see code/b1/setup-node.sh). Run INSIDE an salloc compute job — the login node
# is noexec on NFS and kills heavy procs:
#   salloc -p SCHOOL -w watgpu308 --gres=gpu:2 --cpus-per-task=8 --mem=64G -t 7:00:00
#   bash code/b2/setup.sh
# Serving-only — no udocker / prime-rl / mini-swe-agent. Idempotent; safe to re-run.
set -euo pipefail

# Repo root = two levels up from this script (code/b2/ -> repo root).
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
VLLM="$REPO/code/forks/vllm"
B2=${B2_ROOT:-/u3/u3anand/b2}
export HF_HOME=${HF_HOME:-$B2/models}
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}   # only matters for multi-GPU B7
mkdir -p "$B2"/{models,data,traces,runs}

echo "== [1/5] uv =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1090
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
command -v uv >/dev/null || { echo "uv not on PATH after install"; exit 1; }
uv --version

echo "== [2/5] vLLM fork submodule (pinned in pins.txt) =="
git -C "$REPO" submodule update --init code/forks/vllm

echo "== [3/5] venv (py3.12) at code/b2/.venv =="
cd "$REPO/code/b2"
uv venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== [4/5] vLLM (kernel-free MoE-gate hook → precompiled, no CUDA build) + b2tel =="
VLLM_USE_PRECOMPILED=1 uv pip install -e "$VLLM" --torch-backend=auto
uv pip install -e "$REPO/code/b2"           # numpy + matplotlib + orjson
uv pip install "datasets>=2.18" || true     # make_trace.py (optional; has synthetic fallback)

echo "== [5/5] OLMoE-1B-7B download (BF16; the bring-up model) =="
python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download("allenai/OLMoE-1B-7B-0924-Instruct")
print("OLMoE present under", os.environ.get("HF_HOME"))
PY

echo "== sanity =="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
import b2tel
print("b2tel ok:", b2tel.EXPERT_FIELDS[:6], "...")
from vllm.model_executor.layers.expert_load_logger import maybe_log_expert_load  # noqa: F401
print("vLLM B2 hook import OK")
PY

echo
echo "setup.sh done (venv: code/b2/.venv). Next (see code/b2/RUNBOOK.md):"
echo "  source code/b2/.venv/bin/activate"
echo "  B3 capture: VLLM_B2_EXPERT_LOG_DIR=\$CAP vllm serve allenai/OLMoE-1B-7B-0924-Instruct \\"
echo "    --dtype bfloat16 --enforce-eager --port 8000   (then python -m b2tel.replay ...)"
echo "Pins:"; cat "$REPO/code/b2/pins.txt"
