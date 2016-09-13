#!/usr/bin/env bash

# This is a utility script to pre-generate SSH keys which terraform will copy onto workers.
# Running this will generate the master ssh key at spark.master and worker keys at spark.worker.0,
# spark.worker.1 and so on. Be sure to pre-generate these before running terraform

usage() {
  cat << EOF
usage: $0 <n_workers>...

Options:
  -h        Show help options.
EOF
}

if [[ $# -eq 0 ]] ; then
    usage
    exit 0
fi

n_workers=$(expr $1 - 1)

mkdir -p terraform/ssh-keys

ssh-keygen -t rsa -f terraform/ssh-keys/spark.master -b 4096 -N "" -C "spark.master"
for i in $(seq 0 ${n_workers}); do
    ssh-keygen -t rsa -f terraform/ssh-keys/spark.worker.${i} -b 4096 -N "" -C "spark.worker.${i}"
done
