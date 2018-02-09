#!/usr/bin/env bash

#SBATCH --job-name=qanta
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output qanta.out.%j
#SBATCH --mem 16gb
#SBATCH --ntasks-per-node=4
#SBATCH --error qanta.out.%j
#SBATCH --ntasks=4

srun bash /fs/clip-quiz/qb/slurm-run.sh $1
