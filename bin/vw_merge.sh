#!/usr/bin/env bash

rm -f output/vw_input/$1.sentence.$2.vw_input.gz
rm -f output/vw_input/$1.sentence.$2.vw_input
rm -f output/vw_input/$1.sentence.$2.meta
cat output/vw_input/$1/sentence.$2.vw_input/* | python3 qanta/util/split_feature_meta.py output/vw_input/$1.sentence.$2.vw_input output/vw_input/$1.sentence.$2.meta
cat output/vw_input/$1.sentence.$2.vw_input | gzip > output/vw_input/$1.sentence.$2.vw_input.gz
rm -f output/vw_input/$1.sentence.$2.vw_input
