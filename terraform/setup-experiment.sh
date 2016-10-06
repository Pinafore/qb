#!/bin/bash
source ~/.bashrc
echo 'set -o vi' >> ~/.bashrc
cd $SPARK_HOME
sbin/start-all.sh
cd $QB_ROOT
make clm
python3 setup.py download
export QB_AWS_S3_BUCKET=qanta-experiments
export QB_AWS_S3_NAMESPACE=davis
./checkpoint.py restore preprocess
cd ..
luigid --background
cd $QB_ROOT
git pull origin davis
# luigi --background --module qanta.pipeline.dan RunTFDanExperiment
