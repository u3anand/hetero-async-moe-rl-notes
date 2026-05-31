#!/usr/bin/env bash
# Shared env header sourced by every B1 sbatch/srun. Source after `source activate b1`.
export B1_ROOT=${B1_ROOT:-/u3/u3anand/b1}
export HF_HOME=$B1_ROOT/models
export UDOCKER_DIR=$B1_ROOT/sandbox/.udocker
export NCCL_SOCKET_IFNAME=eth0          # mandatory: NCCL hangs on the veth nodes otherwise
# export NCCL_P2P_DISABLE=1             # uncomment if NCCL still stalls (no NVLink anyway)
export TOKENIZERS_PARALLELISM=false

# Verify GPU index→model map every alloc (ordering not guaranteed); pin tiers by these.
nvidia-smi --query-gpu=index,name --format=csv,noheader
