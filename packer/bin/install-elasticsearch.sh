#!/usr/bin/env bash

set -e

cd ~/dependencies
curl -L -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-5.2.2.tar.gz
tar -xvf elasticsearch-5.2.2.tar.gz

/home/ubuntu/anaconda3/bin/pip install elasticsearch-dsl
