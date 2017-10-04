#!/usr/bin/env bash

set -e

/home/ubuntu/anaconda3/bin/conda install -y pytorch torchvision cuda80 -c soumith
/home/ubuntu/anaconda3/bin/pip install tensorflow
/home/ubuntu/anaconda3/bin/pip install keras
