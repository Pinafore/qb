#!/usr/bin/env bash

rm -f data/vw_input/$1.sentence.$2.vw_input.gz
rm -f data/vw_input/$1.sentence.$2.vw_input
rm -f data/vw_input/$1.sentence.$2.meta
cat data/vw_input/$1/sentence.$2.vw_input/* | python3 qanta/util/split_feature_meta.py data/vw_input/$1.sentence.$2.vw_input data/vw_input/$1.sentence.$2.meta
cat data/vw_input/$1.sentence.$2.vw_input | gzip > data/vw_input/$1.sentence.$2.vw_input.gz
rm -f data/vw_input/$1.sentence.$2.vw_input
