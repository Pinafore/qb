#!/bin/bash

rm -rf /fs/clip-quiz/entilzha/scratch
mkdir -p /fs/clip-quiz/entilzha/scratch

{% for slurm_script in script_list %}
{% if is_scavenger %}
sbatch --account scavenger --partition scavenger --gres {{ gres }} {{ slurm_script }}{% else %}
sbatch {{ slurm_script }}{% endif %}
{% endfor %}

