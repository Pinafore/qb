#!/bin/bash
source ~/qbenv
if [ ! "${QB_ROOT}" ];then
    echo 'Failed to set'
    export QB_ROOT=/ssd-c/qanta/qb
    export PATH=/home/ubuntu/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games
fi
cd $QB_ROOT
make clm
python3 setup.py download
export QB_AWS_S3_BUCKET=dyoshida-checkpoint-us-west-2
export QB_AWS_S3_NAMESPACE=checkpoint
./checkpoint.py restore preprocess
cd ..
luigid --background
cd $QB_ROOT
git checkout davis
