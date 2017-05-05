#!/usr/bin/env bash

cd $QB_ROOT
cd ..
luigid --background

cd $SPARK_HOME
sbin/start-all.sh

cd $QB_ROOT
make clm

bash packer/bin/install-elasticsearch.sh
~/dependencies/elasticsearch-5.2.2/bin/elasticsearch -d

pip install chainer
