#!/usr/bin/env bash

#SBATCH --job-name=qanta
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output qanta.out.%j
#SBATCH --error qanta.out.%j

srun -N 1 -c "slurm-run.sh 0" &
srun -N 1 -c "slurm-run.sh 1" &
srun -N 1 -c "slurm-run.sh 2" &
srun -N 1 -c "slurm-run.sh 3" &
srun -N 1 -c "slurm-run.sh 4" &
wait
