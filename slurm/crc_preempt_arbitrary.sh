#!/bin/bash
##SBATCH --job-name=ndt3
#SBATCH --cluster=gpu
#SBATCH --partition=preempt
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                # default is 4G per core, 256G. But no one else should be using this node.
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=2-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error=slurm_logs/job.%J.err
#SBATCH --output=slurm_logs/job.%J.out

echo $@
hostname
source ~/.bashrc
srun python $@
