#!/usr/bin/env bash

# $1 is the fold

rm -f output/vw_input/$1.vw.gz
rm -f output/vw_input/$1.vw.txt
rm -f output/vw_input/$1.meta
cat output/vw_input/$1.vw/* | python3 qanta/util/split_feature_meta.py output/vw_input/$1.vw.txt output/vw_input/$1.meta
cat output/vw_input/$1.vw.txt | gzip > output/vw_input/$1.vw.gz
rm -f output/vw_input/$1.vw.txt
