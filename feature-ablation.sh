#!/usr/bin/env bash

FOLD=test
WEIGHT=$1
FEATURE=$2

if [ "$FEATURE" = "t" ]
then
    vw --compressed -d data/vw_input/dev.sentence.${WEIGHT}.vw_input.gz --early_terminate 100 -k --ignore ${FEATURE} -q ga -b 24 --loss_function logistic -f data/models/sentence.full.${WEIGHT}.-${FEATURE}.vw
    vw --compressed -t -q ga --ignore ${FEATURE} -d data/vw_input/test.sentence.${WEIGHT}.vw_input.gz -i data/models/sentence.full.${WEIGHT}.-${FEATURE}.vw -p data/results/test/sentence.${WEIGHT}.-${FEATURE}.full.pred
    python3 qanta/reporting/performance.py generate data/results/test/sentence.${WEIGHT}.-${FEATURE}.full.pred data/vw_input/test.sentence.${WEIGHT}.meta data/results/test/sentence.${WEIGHT}.-${FEATURE}.answer.json
elif [ "$FEATURE" = "a" ]
then
    vw --compressed -d data/vw_input/dev.sentence.${WEIGHT}.vw_input.gz --early_terminate 100 -k --ignore ${FEATURE} -q gt -b 24 --loss_function logistic -f data/models/sentence.full.${WEIGHT}.-${FEATURE}.vw
    vw --compressed -t -q gt --ignore ${FEATURE} -d data/vw_input/test.sentence.${WEIGHT}.vw_input.gz -i data/models/sentence.full.${WEIGHT}.-${FEATURE}.vw -p data/results/test/sentence.${WEIGHT}.-${FEATURE}.full.pred
    python3 qanta/reporting/performance.py generate data/results/test/sentence.${WEIGHT}.-${FEATURE}.full.pred data/vw_input/test.sentence.${WEIGHT}.meta data/results/test/sentence.${WEIGHT}.-${FEATURE}.answer.json
else
    vw --compressed -d data/vw_input/dev.sentence.${WEIGHT}.vw_input.gz --early_terminate 100 -k --ignore ${FEATURE} -q gt -q ga -b 24 --loss_function logistic -f data/models/sentence.full.${WEIGHT}.-${FEATURE}.vw
    vw --compressed -t -q ga -q gt --ignore ${FEATURE} -d data/vw_input/test.sentence.${WEIGHT}.vw_input.gz -i data/models/sentence.full.${WEIGHT}.-${FEATURE}.vw -p data/results/test/sentence.${WEIGHT}.-${FEATURE}.full.pred
    python3 qanta/reporting/performance.py generate data/results/test/sentence.${WEIGHT}.-${FEATURE}.full.pred data/vw_input/test.sentence.${WEIGHT}.meta data/results/test/sentence.${WEIGHT}.-${FEATURE}.answer.json
fi
