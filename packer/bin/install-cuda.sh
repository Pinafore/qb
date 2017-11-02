#!/usr/bin/env bash

set -e

# Install CUDA 9
cd ~
sudo apt-get install -y linux-image-extra-`uname -r`
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run -O /home/ubuntu/cuda_9.0.176_384.81_linux-run
sudo sh /home/ubuntu/cuda_9.0.176_384.81_linux-run --silent --driver --toolkit

# Install DNN 7
wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/public/cudnn-9.0-linux-x64-v7.tgz -O /home/ubuntu/cudnn-9.0-linux-x64-v7.tgz
tar zxvf /home/ubuntu/cudnn-9.0-linux-x64-v7.tgz
sudo cp -P /home/ubuntu/cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P /home/ubuntu/cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
sudo apt-get install -y libcupti-dev
cat cuda-dnn-env.sh >> ~/.bashrc

wget https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/public/nccl-repo-ubuntu1604-2.0.5-ga-cuda9.0_3-1_amd64.deb -O /home/ubuntu/nccl-repo-ubuntu1604-2.0.5-ga-cuda9.0_3-1_amd64.deb
sudo dpkg -i /home/ubuntu/nccl-repo-ubuntu1604-2.0.5-ga-cuda9.0_3-1_amd64.deb
