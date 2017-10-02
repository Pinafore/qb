#!/usr/bin/env bash

set -e

# Install Vowpal Wabbit
cd ~/dependencies
git clone git://github.com/JohnLangford/vowpal_wabbit.git
cd vowpal_wabbit
git checkout 8.4.0
make
sudo make install
