#!/usr/bin/env bash

cd $SPARK_HOME
./sbin/start-all.sh

elasticsearch -d

cd $QB_ROOT
luigid --background

