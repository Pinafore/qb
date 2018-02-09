#!/usr/bin/env bash

#SBATCH --job-name=qanta
#SBATCH --qos=gpu
#SBATCH --nodes=3
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --output qanta.out.%j
#SBATCH --mem 16gb
#SBATCH --ntasks-per-node=4
#SBATCH --error qanta.out.%j
#SBATCH --ntasks=12

srun -N 1 --gres=gpu:1 --qos=gpu --partition=gpu bash /fs/clip-quiz/qb/slurm-run.sh 0 &
srun -N 1 --gres=gpu:1 --qos=gpu --partition=gpu bash /fs/clip-quiz/qb/slurm-run.sh 1 &
srun -N 1 --gres=gpu:1 --qos=gpu --partition=gpu bash /fs/clip-quiz/qb/slurm-run.sh 2 &
srun -N 1 --gres=gpu:1 --qos=gpu --partition=gpu bash /fs/clip-quiz/qb/slurm-run.sh 3 &
srun -N 1 --gres=gpu:1 --qos=gpu --partition=gpu bash /fs/clip-quiz/qb/slurm-run.sh 4 &
wait
