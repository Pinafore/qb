#!/usr/bin/env bash

#SBATCH --job-name=qanta
#SBATCH --qos=gpu
#SBATCH --gres=gpu:5

srun -c "slurm-run.sh 0" &
srun -c "slurm-run.sh 1" &
srun -c "slurm-run.sh 2" &
srun -c "slurm-run.sh 3" &
srun -c "slurm-run.sh 4" &
wait
