#!/usr/bin/env bash

set -e

# Create aws credentials directory
mkdir -p /home/ubuntu/.aws

# Create Luigi logging directory
sudo mkdir /var/log/luigi
sudo chown ubuntu /var/log/luigi
sudo mkdir /etc/luigi
sudo mv /home/ubuntu/client.cfg /etc/luigi/client.cfg

# Configure ulimits
sudo mv /home/ubuntu/limits.conf /etc/security/limits.conf
