#!/usr/bin/env bash
# check-node.sh [NODE|all] [--models] — WatGPU availability snapshot.
#   (no arg)            → detailed view of watgpu308 (our node) + can-we-run verdict
#   watgpu408           → detailed view of any node
#   all                 → compact one-line-per-node cluster table
#   --models            → also list physical GPU models (grabs a 1-CPU slot; single-node only)
# Read-only Slurm queries (no allocation unless --models). Run on the login node or:
#   ssh u3anand@watgpu.cs.uwaterloo.ca 'bash -s -- all' < code/b1/slurm/check-node.sh
set -uo pipefail

NODE=watgpu308; MODELS=0; ALL=0
for a in "$@"; do
  case "$a" in
    --models) MODELS=1 ;;
    all|--all) ALL=1 ;;
    watgpu*)  NODE="$a" ;;
  esac
done

# Reachability for us (school_group): 308 is ours via SCHOOL; vision nodes preempt us; rest = preemptible ALL.
reach() { case "$1" in
  watgpu308) echo "ours/SCHOOL" ;;
  watgpu208|watgpu1008) echo "vision/PREEMPTS-us" ;;
  *) echo "ALL/preemptible" ;;
esac; }

if [ "$ALL" = 1 ]; then
  echo "===== WatGPU cluster @ $(date -u +%FT%TZ) ====="
  printf "%-11s %-14s %7s  %9s  %-18s %s\n" NODE STATE GPUfree CPUfree REACH ""
  sresources 2>/dev/null | awk '/^watgpu/{print $1,$2,$4,$6,$12}' | while read -r n st gtot gfree cpuf; do
    flag="🔴"; [ "${gfree:-0}" -gt 0 ] 2>/dev/null && flag="🟢"
    case "$st" in *DRAIN*|*DOWN*|*DRNG*) flag="⚪" ;; esac
    printf "%-11s %-14s %4s/%-2s  %9s  %-18s %s\n" "$n" "$st" "${gfree:-?}" "${gtot:-?}" "${cpuf:-?}" "$(reach "$n")" "$flag"
  done
  echo "(🟢 has free GPU · 🔴 full · ⚪ drain/down · only ours/SCHOOL=308 is reliably holdable; ALL=preemptible; vision preempts us)"
  exit 0
fi

echo "===== $NODE @ $(date -u +%FT%TZ) ($(reach "$NODE")) ====="
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
