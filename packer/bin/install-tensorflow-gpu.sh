#!/usr/bin/env bash

set -e
source ~/.bashrc
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0rc0-cp35-cp35m-linux_x86_64.whl
/home/ubuntu/anaconda3/bin/pip install $TF_BINARY_URL
