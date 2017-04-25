#!/usr/bin/env bash

for var in "$@"
do
    ssh "ubuntu@$var" /home/ubuntu/anaconda3/bin/aws s3 cp s3://entilzha-us-west-2/questions/naqt.db /ssd-c/qanta/qb/data/internal/naqt.db
done