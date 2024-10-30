#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q shared
#SBATCH -J ndt
#SBATCH --mail-user=joelye9@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -A m1266
#SBATCH -t 24:0:0
#SBATCH --error=slurm_logs/job.%J.err
#SBATCH --output=slurm_logs/job.%J.out
#SBATCH -c 32

echo $@
hostname
source ~/.bashrc
srun python $@
