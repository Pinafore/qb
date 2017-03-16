#!/usr/bin/env bash

FEATURE=$1

vw -b 30 -d output/vw_input/dev.vw.txt -k --keep ${FEATURE} --loss_function logistic -f output/models/model.+${FEATURE}.vw
vw -t --keep ${FEATURE} --loss_function logistic -d output/vw_input/test.vw.txt -i output/models/model.+${FEATURE}.vw -p output/predictions/test.+${FEATURE}.pred
python3 qanta/reporting/performance.py generate output/predictions/test.+${FEATURE}.pred output/vw_input/test.meta output/summary/test.+${FEATURE}.json
