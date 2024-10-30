#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -q regular
#SBATCH -J ndt
#SBATCH --mail-user=joelye9@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -A m1266
#SBATCH -t 24:00:0
#SBATCH --error=slurm_logs/job.%J.err
#SBATCH --output=slurm_logs/job.%J.out
#SBATCH --mem 240G
echo $@
hostname
export SLURM_NTASKS_PER_NODE=1
# Using gpu-bind none bc I believe that's wht pytorch lightning expects
source ~/.bashrc
srun python -u run.py slurm_use_scratch=True $@
