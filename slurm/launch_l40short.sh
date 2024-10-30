#!/bin/bash
#SBATCH --job-name=ndt3
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=480G                # default is 4G per core, 256G. But no one else should be using this node.
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=2-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error=slurm_logs/job.%J.err
#SBATCH --output=slurm_logs/job.%J.out

echo "Quick run - 2 days"
# Note we expect 1kh for 300M to take 2 weeks on 8 GPUs, 1 week on 16 GPUs.
echo $@
hostname
source ~/.bashrc
source ~/load_env.sh
srun python -u run.py $@