#!/usr/bin/env bash

set -e

# Install Tensorflow
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc1-cp35-cp35m-linux_x86_64.whl
/home/ubuntu/anaconda3/bin/pip install --upgrade $TF_BINARY_URL || true
