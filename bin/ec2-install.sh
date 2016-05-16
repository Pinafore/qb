#!/usr/bin/env bash

# Note, this script is not tested, it is a list of commands run in some order to get EC2 to work
sudo apt-get update
sudo apt-get install -y git
sudo apt-get install -y build-essential
sudo apt-get install -y r-base
sudo apt-get install -y libboost-program-options-dev
sudo apt-get install -y libboost-python-dev
sudo apt-get install -y libtool
sudo apt-get install -y libboost-all-dev
sudo apt-get install -y cmake
sudo apt-get install -y swig
sudo apt-get install -y python3-dev python-dev
sudo apt-get install -y libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
sudo apt-get install -y default-jre default-jdk

wget https://www.python.org/ftp/python/3.5.1/Python-3.5.1.tar.xz
tar xf Python-3.5.1
cd Python-3.5.1
./configure
make
sudo make install

# Fix python3 binary to point to python3.5
git clone https://github.com/Pinafore/qb.git
mkdir dependencies
cd dependencies
wget https://github.com/JohnLangford/vowpal_wabbit/archive/8.1.1.tar.gz
tar xf 8.1.1.tar.gz
cd vowpal_wabbit-8.1.1/
./autogen
make
sudo make install

# Transfer gorobo and decompress it to ~/dependencies
cd ~/dependencies
tar xf gurobi6.5.0_linux64.tar.gz
cd gurobi650/linux64
sudo python setup.py install

# This is where I rsync everything else over

# Install kenlm
cd ~/dependencies
git clone https://github.com/kpu/kenlm.git
cd kenlm
cmake .
make
sudo make install
sudo python3 setup.py install

# Install pip and packages
cd ~/
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
sudo python get-pip.py
cd qb
sudo pip3 install -requirements.txt
sudo pip install requirements.txt
python3 util/install_nltk_corpora.py

# Set environment variables
echo "export QB_QUESTION_DB=/home/ubuntu/qb/data/non_naqt.db" >> ~/.bashrc
echo "export QB_GUESS_DB=/home/ubuntu/qb/data/guesses.db" >> ~/.bashrc
echo "export QB_SPARK_MASTER=spark://localhost:7077" >> ~/.bashrc
echo "export PYTHONPATH=$PYTHONPATH:/home/ubuntu/"

# Install Spark dependencies
