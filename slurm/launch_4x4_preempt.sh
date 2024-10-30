#!/bin/bash
#SBATCH --job-name=ndt3_4x4
#SBATCH --cluster=gpu
#SBATCH --partition=preempt
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=256G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=4-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error=slurm_logs/job.%J.err
#SBATCH --output=slurm_logs/job.%J.out
#SBATCH -x gpu-n62,gpu-n67 # dead

echo "4 GPU run on 4 node for 4 days. For biggest runs."
# Note we expect 1kh for 300M to take 2 weeks on 8 GPUs, 1 week on 16 GPUs.
echo $@
hostname

source ~/.bashrc
source ~/load_env.sh
srun python -u run.py nodes=4 slurm_use_scratch=True $@
crc-job-stats.py
