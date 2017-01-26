#!/usr/bin/env bash

set -e

# Install CUDA 8
cd ~
sudo apt-get install -y linux-image-extra-`uname -r`
wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run -O /home/ubuntu/cuda_8.0.44_linux-run
sudo sh /home/ubuntu/cuda_8.0.44_linux-run --silent --driver --toolkit

# Install DNN 5
wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/public/cudnn-8.0-linux-x64-v5.1.tgz -O /home/ubuntu/cudnn-8.0-linux-x64-v5.1.tgz
tar zxvf /home/ubuntu/cudnn-8.0-linux-x64-v5.1.tgz
sudo cp -P /home/ubuntu/cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P /home/ubuntu/cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
sudo apt-get install -y libcupti-dev
cat cuda-dnn-env.sh >> ~/.bashrc
