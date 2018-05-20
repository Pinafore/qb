#!/usr/bin/env bash

set -e

# This script isn't meant to be runnable, more a documentation of what commands to run

# Convert old database to current question format
sqlite3 -header -csv data/external/non_naqt.db < bin/non_naqt_to_csv.sql > data/external/non_naqt.csv
./cli.py nonnaqt_to_json data/external/non_naqt.csv data/external/datasets/

# Make sure torchtext uses it
mkdir -p .data/quizbowl
cp data/external/datasets/qanta.torchtext* .data/quizbowl

# Convert adversarial interface questions to current question format
./cli.py adversarial_to_json data/external/datasets/final_round_adversarial.json data/external/datasets

# The models are already pretrained, just need to remove expo targets
rm output/guesser/qanta.guesser*/0/guesser_report_expo.pickle
rm output/guesser/qanta.guesser*/0/guesses_*_expo.pickle

# Now run the reports
luigi --local-scheduler --module qanta.pipeline.guesser AllGuesserReports

# Last thing is generate the plots
./figures.py guesser output/report

# That generates dev set plots, this generates test set plots
./figures.py guesser --use-test output/report
