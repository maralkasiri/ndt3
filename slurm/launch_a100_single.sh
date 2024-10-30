#!/bin/bash
#SBATCH --job-name=ndt3_single
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --account jcollinger
#SBATCH -c 16
#SBATCH --mem=64G                # default is 4G per core, 256G. But no one else should be using this node.
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=1-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error=slurm_logs/job.%J.err
#SBATCH --output=slurm_logs/job.%J.out

echo "Very short"
# Note we expect 1kh for 300M to take 2 weeks on 8 GPUs, 1 week on 16 GPUs.
echo $@
hostname
source ~/.bashrc
source ~/load_env.sh
srun python -u run.py $@