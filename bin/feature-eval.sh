#!/usr/bin/env bash

FILE=$1
FEATURE=$2

vw -b 30 -d output/vw_input/dev.vw.txt -k --keep ${FEATURE} --loss_function logistic -f output/models/model.+${FILE}.vw
vw -t --keep ${FEATURE} --loss_function logistic -d output/vw_input/test.vw.txt -i output/models/model.+${FILE}.vw -p output/predictions/test.+${FILE}.pred
python3 qanta/reporting/performance.py generate output/predictions/test.+${FILE}.pred output/vw_input/test.meta output/summary/test.+${FILE}.json
