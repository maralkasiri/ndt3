#!/bin/bash
#SBATCH --job-name=ndt3_single
#SBATCH --cluster=gpu
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G                # default is 4G per core, 256G. But no one else should be using this node.
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=3-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error=slurm_logs/job.%J.err
#SBATCH --output=slurm_logs/job.%J.out
#SBATCH --mail-user=joelye9@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --signal=SIGUSR1@90 # Timeout signal to be caught for requeue in lightning

echo "Run that copies data dir to local scratch, runs, then copies back."
echo "Specifically only runs copies _data_"
echo "Needed as NDT3's ft-ing load on global storage is too high. (Received complaints, not sure how to benchmark)"

echo $@
echo hostname
source ~/.bashrc
source ~/load_env.sh
srun python -u run.py slurm_use_scratch=True $@