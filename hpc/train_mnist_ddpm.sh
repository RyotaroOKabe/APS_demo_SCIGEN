#!/bin/bash
#SBATCH -J train_mnist_ddpm
#SBATCH -A m4768
#SBATCH -C gpu
#SBATCH --gpus 1
#SBATCH -q premium
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -o ./slurm/mnist_ddpm_%j.log
#SBATCH -e ./slurm/mnist_ddpm_%j.err

set -euo pipefail
mkdir -p ./slurm

# --- ENVIRONMENT ---
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

source ~/.bashrc
conda activate mat_gen_all

# Diagnostics
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURMD_NODENAME}"
python --version
nvidia-smi

cd /pscratch/sd/r/ryotaro/data/generative/APS_demo_SCIGEN

# Train MNIST DDPM for 20 epochs (~5 min on 1 A100)
python scripts/train_mnist_ddpm.py \
    --n_epoch 20 \
    --batch_size 256 \
    --n_feat 128 \
    --output models/mnist_ddpm/ddpm_mnist_20ep.pth \
    --data_dir ./data

echo "Training complete."
ls -lh models/mnist_ddpm/
