#!/usr/bin/env bash
# check-node.sh [NODE] [--models] — live usage snapshot of a WatGPU node + a
# "can we run our workload" verdict. Read-only (Slurm queries only; no allocation
# unless --models is passed, which grabs a 1-CPU slot to read GPU models).
#
# Run on the login node, or from the laptop:
#   ssh u3anand@watgpu.cs.uwaterloo.ca 'bash -s' < code/b1/slurm/check-node.sh
#   ssh u3anand@watgpu.cs.uwaterloo.ca 'bash -s -- watgpu308 --models' < code/b1/slurm/check-node.sh
set -uo pipefail

NODE=watgpu308; MODELS=0
for a in "$@"; do
  case "$a" in
    --models) MODELS=1 ;;
    watgpu*)  NODE="$a" ;;
  esac
done

echo "===== $NODE @ $(date -u +%FT%TZ) ====="

# GPU/CPU/mem from sresources (main row; sub-rows are per-sponsor GRES)
read -r st gtot galloc gfree memf cput cpualloc cpufree < <(
  sresources 2>/dev/null | awk -v n="$NODE" '$1==n{print $2,$4,$5,$6,$9,$10,$11,$12; exit}'
)
gfree=${gfree:-0}; cpufree=${cpufree:-0}; gtot=${gtot:-0}
echo "state=${st:-?}   GPU: ${gfree}/${gtot} free (alloc ${galloc:-?})   CPU: ${cpufree}/${cput:-?} free   MEMfree=${memf:-?}"

echo "-- jobs on $NODE --"
squeue -w "$NODE" -o '%.10i %.9P %.9u %.2t %.11M %.6b %R' 2>/dev/null || echo "(none)"

if [ "$MODELS" = 1 ]; then
  echo "-- physical GPU models (/proc) --"
  timeout 40 srun -p ALL --immediate=20 -w "$NODE" --cpus-per-task=1 --mem=1G -t 00:01:00 \
    bash -c 'grep -h Model /proc/driver/nvidia/gpus/*/information | sed "s/Model:[[:space:]]*//" | sort | uniq -c' \
    2>/dev/null || echo "(busy — could not grab a slot)"
fi

echo "-- verdict (workload: 3-config = 4 GPU/config interactive, or 8 GPU --exclusive run) --"
case "${st:-}" in
  *DRAIN*|*DOWN*|*DRNG*|*NPC*) echo "UNAVAILABLE 🔴  state=$st" ;;
  *)
    if   [ "$gfree" -ge 8 ] && [ "$cpufree" -ge 24 ]; then
      echo "READY ✅  whole node ~free → --exclusive 3-config sbatch can land now"
    elif [ "$gfree" -ge 4 ]; then
      echo "PARTIAL 🟡  $gfree GPU free → one-config interactive spike (4 GPU) OK if tier mix fits; not enough for --exclusive"
    elif [ "$gfree" -ge 1 ]; then
      echo "LIMITED 🟠  $gfree GPU free → recon / sandbox / single-GPU inference spike only"
    else
      echo "BUSY 🔴  0 GPU free → salloc would queue (SCHOOL jobs also risk preemption)"
    fi ;;
esac
