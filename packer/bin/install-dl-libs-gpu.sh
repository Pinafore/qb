#!/usr/bin/env bash

set -e

/home/ubuntu/anaconda3/bin/conda -y install pytorch torchvision cuda80 -c soumith
/home/ubuntu/anaconda3/bin/pip install tensorflow-gpu
/home/ubuntu/anaconda3/bin/pip install keras
