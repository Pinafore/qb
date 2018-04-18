#!/usr/bin/env bash

#SBATCH --job-name=qanta-guesser
#SBATCH --partition=dpart
#SBATCH --qos=batch
#SBATCH --mem-per-cpu=1g
#SBATCH --chdir=/fs/clip-quiz/entilzha/qb
#SBATCH --output=/fs/clip-quiz/entilzha/slurm-out/slurm-%A_%a.out
#SBATCH --error=/fs/clip-quiz/entilzha/slurm-err/slurm-%A_%a.out
#SBATCH --time=1-00:00:00
#SBATCH --dependency=singleton

set -x

srun python slack.py slurm "qanta-guesser jobs completed"
