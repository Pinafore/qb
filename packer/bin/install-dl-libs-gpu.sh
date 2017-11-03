#!/usr/bin/env bash

set -e

/home/ubuntu/anaconda3/bin/pip install tensorflow-gpu
/home/ubuntu/anaconda3/bin/pip install keras

cd /home/ubuntu
git clone --recursive https://github.com/pytorch/pytorch.git
cd /home/ubuntu/pytorch
git checkout 66d24c5
git submodule update --recursive
/home/ubuntu/anaconda3/bin/python setup.py install
