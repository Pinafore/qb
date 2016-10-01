#!/usr/bin/env bash

FOLD=test
WEIGHT=$1
FEATURE=$2

if [ "$FEATURE" = "a" ]
then
    vw --compressed -d data/vw_input/dev.sentence.${WEIGHT}.vw_input.gz --early_terminate 100 -k --keep g --keep a -q ga -b 28 --loss_function logistic -f data/models/sentence.full.${WEIGHT}.a.vw
    vw --compressed -t --keep g --keep a -q ga -d data/vw_input/test.sentence.${WEIGHT}.vw_input.gz -i data/models/sentence.full.${WEIGHT}.a.vw -p data/results/test/sentence.${WEIGHT}.a.full.pred
    python3 qanta/reporting/performance.py generate data/results/test/sentence.${WEIGHT}.a.full.pred data/vw_input/test.sentence.${WEIGHT}.meta data/results/test/sentence.${WEIGHT}.a.answer.json
else
    vw --compressed -d data/vw_input/dev.sentence.${WEIGHT}.vw_input.gz --early_terminate 100 -k --keep g --keep ${FEATURE} -q ga -b 28 --loss_function logistic -f data/models/sentence.full.${WEIGHT}.${FEATURE}.vw
    vw --compressed -t --keep g --keep ${FEATURE} -q ga -d data/vw_input/test.sentence.${WEIGHT}.vw_input.gz -i data/models/sentence.full.${WEIGHT}.${FEATURE}.vw -p data/results/test/sentence.${WEIGHT}.${FEATURE}.full.pred
    python3 qanta/reporting/performance.py generate data/results/test/sentence.${WEIGHT}.${FEATURE}.full.pred data/vw_input/test.sentence.${WEIGHT}.meta data/results/test/sentence.${WEIGHT}.${FEATURE}.answer.json
fi
