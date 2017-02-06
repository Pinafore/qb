#!/usr/bin/env bash

sudo fallocate -l 10G /ssd-c/swapfile
sudo chmod 600 /ssd-c/swapfile
sudo mkswap /ssd-c/swapfile
sudo swapon /ssd-c/swapfile
sudo sysctl vm.swappiness=10
