#!/usr/bin/env bash

python3 setup.py download

mkdir -p data/external
aws s3 cp s3://pinafore/public/wikipedia.tar.gz /tmp/wikipedia.tar.gz
tar -xvzf /tmp/wikipedia.tar.gz -C data/external
rm /tmp/wikipedia.tar.gz

aws s3 cp s3://pinafore/public/Wikifier2013.zip /tmp/Wikifier2013.zip
unzip /tmp/Wikifier2013.zip -d data/external
rm /tmp/Wikifier2013.zip
