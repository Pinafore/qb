#!/usr/bin/env bash

/home/ubuntu/anaconda3/bin/luigid --background

cd /home/ubuntu/dependencies/spark-2.2.0-bin-hadoop2.7
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/ubuntu/cuda:/usr/local/cuda/lib64:/usr/local/extras/CUPTI/lib64" >> conf/spark-env.sh
echo "export CUDA_HOME=/usr/local/cuda" >> conf/spark-env.sh

cd /ssd-c/qanta/qb
bash packer/bin/install-elasticsearch.sh
/home/ubuntu/dependencies/elasticsearch-5.6.2/bin/elasticsearch -d
