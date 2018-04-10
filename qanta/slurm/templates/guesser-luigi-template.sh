#!/bin/bash
#SBATCH --qos=batch
#SBATCH --mem-per-cpu=12g
#SBATCH --chdir=/fs/clip-quiz/entilzha/qb
#SBATCH --output=/fs/clip-quiz/entilzha/slurm-out/slurm-%A_%a.out
#SBATCH --time=23:59:59

set -x

srun luigi --module qanta.pipeline.guesser --workers 1 GuesserReport \
  --guesser-module {{ gs.guesser_module }} --guesser-class {{ gs.guesser_class}} \
  --dependency-module {{ gs.dependency_module }} --dependency-class {{ gs.dependency_class }} \
  --config-num {{ gs.config_num }}
