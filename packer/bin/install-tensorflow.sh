#!/usr/bin/env bash

set -e

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0rc0-cp36-cp36m-linux_x86_64.whl
/home/ubuntu/anaconda3/bin/pip install $TF_BINARY_URL
