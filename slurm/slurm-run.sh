#!/usr/bin/env bash

mkdir -p /scratch0/qanta/"$1"/
cp -r -u /fs/clip-quiz/qb /scratch0/qanta/"$1"/
rm -rf /scratch0/qanta/"$1"/qb/output/guesser
ls /scratch0/qanta/"$1"/qb
nvidia-smi