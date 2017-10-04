#!/usr/bin/env bash

set -e

# Install Apache Spark
cd /home/ubuntu
wget https://d3kbcqa49mib13.cloudfront.net/spark-2.2.0-bin-hadoop2.7.tgz
mkdir -p /home/ubuntu/dependencies/
tar xzf spark-2.2.0-bin-hadoop2.7.tgz -C /home/ubuntu/dependencies
rm spark-2.2.0-bin-hadoop2.7.tgz
mv /home/ubuntu/spark-defaults.conf /home/ubuntu/dependencies/spark-2.2.0-bin-hadoop2.7/conf
echo "export PYSPARK_PYTHON=/home/ubuntu/anaconda3/bin/python3" >> ~/.bashrc
