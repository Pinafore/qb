#!/usr/bin/env bash

#SBATCH --job-name=qanta-guesser
#SBATCH --partition={{ partition }}
#SBATCH --qos={{ qos }}
#SBATCH --mem-per-cpu={{ mem_per_cpu }}
#SBATCH --chdir=/fs/clip-quiz/entilzha/qb
#SBATCH --output=/fs/clip-quiz/entilzha/slurm-logs/stdout-%A_%a.out
#SBATCH --error=/fs/clip-quiz/entilzha/slurm-logs/stderr-%A_%a.out
#SBATCH --time={{ max_time }}
{% if gres %}
#SBATCH --gres={{ gres }}
{% endif %}
{% if cpus_per_task %}#SBATCH --cpus-per-task={{ cpus_per_task }}
{% endif %}

set -x

srun luigi --module qanta.pipeline.guesser --workers 1 {{ task }} \
  --guesser-module {{ guesser_module }} --guesser-class {{ guesser_class}} \
  --dependency-module {{ dependency_module }} --dependency-class {{ dependency_class }} \
  --config-num {{ config_num }}
