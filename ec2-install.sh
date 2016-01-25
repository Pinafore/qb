sudo apt-get update
sudo apt-get install -y git
sudo apt-get install -y build-essential
sudo apt-get install -y r-base
wget https://www.python.org/ftp/python/3.5.1/Python-3.5.1.tar.xz
cd Python-3.5.1
./configure
make
sudo make install

# Fix python3 binary to point to python3.5
