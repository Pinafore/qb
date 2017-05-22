#!/usr/bin/env bash

set -e

cd /ssd-c/qanta/qb

/home/ubuntu/anaconda3/bin/python setup.py download

mkdir -p data/external/wikipedia
/home/ubuntu/anaconda3/bin/aws s3 cp s3://pinafore-us-west-2/public/wiki_redirects.csv data/external/wikipedia/all_wiki_redirects.csv
/home/ubuntu/anaconda3/bin/aws s3 cp s3://pinafore-us-west-2/public/wikipedia-normed.tar.gz /tmp/wikipedia.tar.gz
tar -xvzf /tmp/wikipedia.tar.gz -C data/external/pages
touch data/external/wikipedia/pages/wikipedia_page_SUCCESS
rm /tmp/wikipedia.tar.gz

# /home/ubuntu/anaconda3/bin/aws s3 cp s3://pinafore-us-west-2/public/Wikifier2013.zip /tmp/Wikifier2013.zip
# unzip /tmp/Wikifier2013.zip -d data/external
# rm /tmp/Wikifier2013.zip

mkdir -p data/external/deep
/home/ubuntu/anaconda3/bin/aws s3 cp s3://pinafore-us-west-2/public/glove.6B.300d.txt data/external/deep/glove.6B.300d.txt
