#!/usr/bin/env bash

set -e

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y build-essential cmake swig python-software-properties
sudo apt-get install -y git wget vim tmux unzip
sudo apt-get install -y libboost-program-options-dev libboost-python-dev libtool libboost-all-dev
sudo apt-get install -y liblzma-dev libpq-dev liblz4-tool
sudo apt-get install -y default-jre default-jdk

# Install Docker
sudo apt-get install -y apt-transport-https ca-certificates
sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
echo "deb https://apt.dockerproject.org/repo ubuntu-trusty main" | sudo tee --append /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get purge lxc-docker
sudo apt-get install -y linux-image-extra-$(uname -r) apparmor
sudo apt-get install -y docker-engine
sudo usermod -aG docker ubuntu

sudo add-apt-repository ppa:keithw/mosh
sudo apt-get update
sudo apt-get install -y mosh

# Install LaTeX for reporting
sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-extra texlive-fonts-recommended
