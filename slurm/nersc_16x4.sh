#!/bin/bash
#SBATCH -N 16
#SBATCH -C gpu
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH -q regular
#SBATCH -J ndt
#SBATCH --mail-user=joelye9@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -A m1266
#SBATCH -t 24:00:00
#SBATCH --error=slurm_logs/job.%J.err
#SBATCH --output=slurm_logs/job.%J.out
#SBATCH --signal=SIGUSR1@90 # Timeout signal to be caught for requeue in lightning
echo "1 days, 64 GPU"
echo $@
hostname
export SLURM_NTASKS_PER_NODE=4
# Using gpu-bind none bc I believe that's what pytorch lightning expects
source ~/.bashrc
srun python -u run.py nodes=16 slurm_use_scratch=True $@
