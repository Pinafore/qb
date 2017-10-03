#!/usr/bin/env bash

set -e

# Install Apache Spark
wget https://d3kbcqa49mib13.cloudfront.net/spark-2.2.0-bin-hadoop2.7.tgz
tar xzf spark-2.2.0-bin-hadoop2.7.tgz
rm spark-2.2.0-bin-hadoop2.7.tgz
mv spark-2.2.0-bin-hadoop2.7 /home/ubuntu/dependencies
echo "export PYSPARK_PYTHON=/home/ubuntu/anaconda3/bin/python3" >> ~/.bashrc
