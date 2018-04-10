#!/bin/bash

{% for slurm_script in script_list %}
sbatch {{ slurm_script }}
{% endfor %}

