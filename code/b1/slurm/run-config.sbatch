#!/usr/bin/env bash
#SBATCH --job-name=b1-run
#SBATCH --partition=SCHOOL
#SBATCH --nodelist=watgpu308
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@180
#SBATCH -o %x-%j.out -e %x-%j.err
#SBATCH --mail-type=ALL
# Run one config of the 3-config comparison. Usage:
#   sbatch --export=CONFIG=homo-fast run-config.sbatch
# CONFIG ∈ {homo-fast, homo-slow, hetero}; selects configs/infer-$CONFIG.toml.
set -euo pipefail
CONFIG=${CONFIG:?set CONFIG=homo-fast|homo-slow|hetero}
TS=$(date -u +%Y%m%dT%H%M%S)
RUN=$B1_ROOT/runs/$CONFIG/$TS

source activate b1
HERE=$(dirname "$(readlink -f "$0")")
source "$HERE/_env.sh"
mkdir -p "$RUN"

# --- preemption: SIGUSR1 → checkpoint + exit cleanly; --requeue restarts → resume from ckpt
cleanup() { echo "SIGUSR1: checkpointing $RUN"; touch "$RUN/.preempt_request"; }
trap cleanup SIGUSR1

# --- per-tier GPU pinning is set inside the infer TOML / launch wrapper (verify indices first).
# --- system sidecar
python -m b1tel.sampler --run-dir "$RUN" --interval 1.0 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

CFG=$HERE/../configs
RESUME=""; [ -d "$RUN/../"*"/ckpt" ] && RESUME="--resume_from_latest"   # placeholder; wire to prime-rl

# --- launch prime-rl (trainer + orchestrator + inference) for this config
uv run rl \
  --trainer @ "$CFG/train.toml" \
  --orchestrator @ "$CFG/orch.toml" \
  --inference @ "$CFG/infer-$CONFIG.toml" \
  --output-dir "$RUN" $RESUME &
RL=$!
wait $RL
