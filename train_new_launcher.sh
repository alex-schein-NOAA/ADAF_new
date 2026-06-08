#!/bin/bash

#SBATCH --ntasks-per-node=1          # BACK TO: one launcher task per node
#SBATCH --cpus-per-task=4 #24 
#SBATCH --gres=gpu:2                 # 2 GPUs per node

echo "Starting job"

# --- Threading: 2 ranks × 2 threads = 4 CPUs/node ---
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# --- NCCL / rendezvous ---
#export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Network (tweak iface if needed, e.g., ib0, enp175s0f0np0)
# export NCCL_SOCKET_IFNAME=^lo,docker0

# Rendezvous (shared by all nodes)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID
export RDZV_BACKEND=c10d
export RDZV_ENDPOINT=${MASTER_ADDR}:${MASTER_PORT}
export RDZV_ID=$SLURM_JOB_ID
##Added from Raj's code
# export WORLD_SIZE=4 #Changed from 8 to 4, for use on 4 nodes #(2025-10-27) this doesn't match the world_size in the Python printout - that's 8 (i.e. # of GPUs) rather than the number of nodes. Maybe for future runs, comment this out if it's causing issues?

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_NODEID=$SLURM_NODEID / SLURM_NNODES=$SLURM_NNODES"

echo "starting at $(date)"
startTime=$(date +%s)
## Added from Raj's code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


###############

#cd /scratch3/BMC/wrfruc/aschein
#cd /scratch3/SYSADMIN/nesccmgmt/Ron.Millikan/devl/alex
echo $PWD

module load python
echo 'Modules loaded'

source /scratch3/BMC/wrfruc/aschein/miniconda/etc/profile.d/conda.sh
conda activate ADAF_environment_pip

echo "After Python load: CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

###############

# --- Quick sanity check on *every* node about GPU visibility/binding ---
#NOT adding the arguments from Raj's code, at least not yet
srun --ntasks-per-node=2 --mpi=none \
     --gres=gpu:2 \
     bash -lc 'echo "Host: $(hostname)"; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L || true'

# --- Launch: one torchrun per node; each spawns 2 ranks (1 per GPU) ---

srun --ntasks-per-node=2 --mpi=none \
     --gres=gpu:2 \
     --gpu-bind=map_gpu:0,1 \
#   torchrun \
    python -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc_per_node=2 \
    --node_rank="${NODE_RANK}" \
    --rdzv_backend="${RDZV_BACKEND}" \
    --rdzv_endpoint="${RDZV_ENDPOINT}" \
    --rdzv_id="${RDZV_ID}" \
    /scratch3/BMC/wrfruc/aschein/ADAF_new/train_new.py

stopTime=$(date +%s)
echo "runTime=$((stopTime-startTime))"