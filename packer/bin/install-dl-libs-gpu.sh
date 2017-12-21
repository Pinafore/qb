#!/usr/bin/env bash

set -e

/home/ubuntu/anaconda3/bin/pip install tensorflow-gpu
/home/ubuntu/anaconda3/bin/pip install keras
/home/ubuntu/anaconda3/bin/pip install pip install git+https://github.com/salesforce/cove.git
/home/ubuntu/anaconda3/bin/conda install -y pytorch torchvision cuda90 -c pytorch
