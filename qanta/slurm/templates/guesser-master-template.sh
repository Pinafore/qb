#!/bin/bash

rm -rf /fs/clip-quiz/entilzha/scratch
mkdir -p /fs/clip-quiz/entilzha/scratch

{% for slurm_script in script_list %}
sbatch {{ slurm_script }}
{% endfor %}

