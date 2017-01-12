#!/usr/bin/env bash

set -e

cd /ssd-c/qanta/qb

/home/ubuntu/anaconda3/bin/python setup.py download

mkdir -p data/external
/home/ubuntu/anaconda3/bin/aws s3 cp s3://pinafore-us-west-2/public/wikipedia.tar.gz /tmp/wikipedia.tar.gz
tar -xvzf /tmp/wikipedia.tar.gz -C data/external
touch data/external/wikipedia_SUCESS
rm /tmp/wikipedia.tar.gz

/home/ubuntu/anaconda3/bin/aws s3 cp s3://pinafore-us-west-2/public/Wikifier2013.zip /tmp/Wikifier2013.zip
unzip /tmp/Wikifier2013.zip -d data/external
rm /tmp/Wikifier2013.zip

mkdir -p data/external/deep
/home/ubuntu/anaconda3/bin/aws s3 cp s3://pinafore-us-west-2/public/glove.840B.300d.txt data/external/deep/glove.6B.300d.txt
