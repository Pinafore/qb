#!/usr/bin/env bash

set -e

# Install Python 3.5 and the Scipy Stack
wget https://repo.continuum.io/archive/Anaconda3-4.3.0-Linux-x86_64.sh
bash Anaconda3-4.3.0-Linux-x86_64.sh -b
rm Anaconda3-4.3.0-Linux-x86_64.sh
echo "export PATH=/home/ubuntu/anaconda3/bin:$PATH" >> ~/.bashrc
echo "export PYTHONPATH=$PYTHONPATH:/home/ubuntu/dependencies/spark-2.1.0-bin-hadoop2.7/python" >> ~/.bashrc
echo "export SPARK_HOME=/home/ubuntu/dependencies/spark-2.1.0-bin-hadoop2.7" >> ~/.bashrc
cat /home/ubuntu/aws-qb-env.sh >> ~/.bashrc

# Install Python dependencies
/home/ubuntu/anaconda3/bin/pip install -r requirements.txt
