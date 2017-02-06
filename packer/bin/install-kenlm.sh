#!/usr/bin/env bash

set -e

# Install KenLM
mkdir ~/dependencies
cd ~/dependencies
git clone https://github.com/kpu/kenlm.git
cd kenlm
cmake .
make
sudo make install
/home/ubuntu/anaconda3/bin/python3 setup.py install
