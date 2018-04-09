#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --qos=batch
#SBATCH --mem-per-cpu=16g
#SBATCH --chdir=/fs/clip-quiz/entilzha/qb
#SBATCH --output=/fs/clip-quiz/entilzha/slurm-out/slurm-%A_%a.out

set -x

srun luigi --module qanta.pipeline.guesser --workers 1 AllSingleGuesserReports