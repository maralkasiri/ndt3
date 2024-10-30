#!/bin/bash
#SBATCH --job-name=ndt3
#SBATCH --nodes=1
#SBATCH --gres gpu:1
#SBATCH -p gpu
#SBATCH -c 6
#SBATCH -t 12:00:00
#SBATCH --mem 30G
#SBATCH -x mind-1-1,mind-1-3,mind-1-5,mind-1-7,mind-1-9,mind-1-11,mind-1-13,mind-1-23,mind-1-26,mind-1-28,mind-1-30,mind-0-3,mind-0-25
#SBATCH --output=slurm_logs/job.%J.out

echo $@
hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
# source ~/load_env.sh
mamba activate torch2
python -u run.py $@

