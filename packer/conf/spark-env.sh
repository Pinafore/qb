#!/usr/bin/env bash

export PYSPARK_PYTHON=/home/ubuntu/anaconda3/bin/python
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/ubuntu/cuda:/usr/local/cuda/lib64:/usr/local/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda