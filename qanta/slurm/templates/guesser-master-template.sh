#!/bin/bash

rm -rf /fs/clip-quiz/entilzha/scratch
mkdir -p /fs/clip-quiz/entilzha/scratch

{% for slurm_script in script_list %}
{% if is_scavenger %}
sbatch --job-name="qanta-guesser {{ slurm_script }}" --chdir "/fs/clip-quiz/entilzha/qb" --mem-per-cpu 4g --output "/fs/clip-quiz/entilzha/slurm-logs/stdout-%A_%a.out" --error "/fs/clip-quiz/entilzha/slurm-logs/stderr-%A_%a.out" --time "1-00:00:00" --account scavenger --partition scavenger --gres {{ gres }} {{ slurm_script }}{% else %}
sbatch {{ slurm_script }}{% endif %}
{% endfor %}

