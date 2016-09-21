#!/usr/bin/env bash

set -e

# Install XGBoost
cd ~/dependencies
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
git checkout v0.60
make -j4
cd python-package
/home/ubuntu/anaconda3/bin/python setup.py install
