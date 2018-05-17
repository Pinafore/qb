#!/usr/bin/env bash

set -e
sqlite3 -header -csv data/external/non_naqt.db < bin/non_naqt_to_csv.sql > data/external/non_naqt.csv
./cli.py data/external/non_naqt.csv data/external/datasets/
mkdir -p .data/quizbowl
cp data/external/datasets/qanta.torchtext* .data/quizbowl
./cli.py data/external/first_round_adversarial.txt data/external/datasets
