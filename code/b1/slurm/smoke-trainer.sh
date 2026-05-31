#!/usr/bin/env bash
# B1 Tier-1 smoke run — trainer-only, fake data, 1 GPU. Proves step-stream telemetry +
# the torch CUDA trace + the sampler sidecar come out on real hardware (OLMoE-1B-7B).
# Run inside the same salloc as setup-node.sh (after it has completed):
#   bash code/b1/slurm/smoke-trainer.sh
# Validates 3 things: step.jsonl(t_train), system.jsonl(per-GPU), trace_0.json.gz.
set -euo pipefail

HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
FORK="$REPO/code/forks/prime-rl"
B1=${B1_ROOT:-/u3/u3anand/b1}

# shellcheck disable=SC1090
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
export HF_HOME=${HF_HOME:-$B1/models}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # one GPU; verify index via `nvidia-smi -L`

TS=$(date -u +%Y%m%dT%H%M%S)
RUN="$B1/runs/smoke/$TS"
mkdir -p "$RUN/trace"
echo "RUN=$RUN  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

cd "$FORK"

# --- system sidecar: 1 Hz nvidia-smi → $RUN/telemetry/system.jsonl (same run-dir as the trainer) ---
uv run python -m b1tel.sampler --run-dir "$RUN" --interval 1.0 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

# --- trainer-only, single GPU, fake data, trace on. torchrun standalone (the rl entrypoint
#     launches the trainer under torchrun; replicate it for one local rank). ---
uv run torchrun --standalone --nproc-per-node=1 \
  -m prime_rl.trainer.rl.train \
  @ "$REPO/code/b1/configs/smoke-train.toml" \
  --output-dir "$RUN" \
  --trace-path "$RUN/trace"

# give the sampler a tick to flush, then stop it
sleep 2; kill $SAMPLER 2>/dev/null || true; trap - EXIT

echo "== validation =="
uv run python - "$RUN" <<'PY'
import gzip, json, sys
from pathlib import Path

run = Path(sys.argv[1]); tel = run / "telemetry"
ok = True

def check(name, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    ok = ok and cond

# (a) step.jsonl with t_train
step = tel / "step.jsonl"
rows = [json.loads(l) for l in step.read_text().splitlines() if l.strip()] if step.exists() else []
have_ttrain = bool(rows) and all("t_train" in r for r in rows)
check("step.jsonl has t_train", have_ttrain, f"{len(rows)} rows")
if rows:
    bcast = sum(1 for r in rows if r.get("broadcast"))
    print(f"        (broadcast sub-record on {bcast}/{len(rows)} rows — absent is EXPECTED trainer-only/Q4 is Tier 2)")

# (b) system.jsonl per-GPU rows
sysf = tel / "system.jsonl"
srows = [json.loads(l) for l in sysf.read_text().splitlines() if l.strip()] if sysf.exists() else []
need = {"gpu_index", "sm_active", "dram_active"}
have_sys = bool(srows) and all(need <= set(r) for r in srows)
check("system.jsonl per-GPU sm/dram rows", have_sys, f"{len(srows)} rows")

# (c) trace_0.json.gz exists and opens
traces = list((run / "trace").glob("trace_*.json.gz"))
trace_ok = False
if traces:
    try:
        with gzip.open(traces[0]) as f:
            json.load(f)
        trace_ok = True
    except Exception as e:
        print(f"        trace open error: {e!r}")
check("CUDA trace_*.json.gz opens", trace_ok, traces[0].name if traces else "no trace file")

print(f"\nSMOKE {'OK ✅' if ok else 'FAILED ❌'}  → {run}")
sys.exit(0 if ok else 1)
PY
