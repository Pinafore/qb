#!/usr/bin/env bash

#SBATCH --job-name=qanta
#SBATCH --qos=gpu
#SBATCH --nodes=5
#SBATCH --gres=gpu:5
#SBATCH --partition=gpu

srun -N 1 -c "slurm-run.sh 0" &
srun -N 1 -c "slurm-run.sh 1" &
srun -N 1 -c "slurm-run.sh 2" &
srun -N 1 -c "slurm-run.sh 3" &
srun -N 1 -c "slurm-run.sh 4" &
wait
