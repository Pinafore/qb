#!/usr/bin/env bash

set -e

cd /home/ubuntu/dependencies
curl -L -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-5.6.2.tar.gz
tar -xvf elasticsearch-5.6.2.tar.gz
