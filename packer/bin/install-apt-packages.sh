#!/usr/bin/env bash

set -e

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get remove -y python-setuptools
sudo apt-get install -y build-essential cmake swig python-software-properties
sudo apt-get install -y git wget vim tmux unzip
sudo apt-get install -y libboost-program-options-dev libboost-python-dev libtool libboost-all-dev
sudo apt-get install -y liblzma-dev libpq-dev liblz4-tool
sudo apt-get install -y default-jre default-jdk

# Install LaTeX for reporting
sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-extra texlive-fonts-recommended
sudo apt-get install -y pdftk
