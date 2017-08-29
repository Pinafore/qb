#!/usr/bin/env bash

set -e
PS1=non-empty source ~/.bashrc
bash terraform/aws-downloads.sh