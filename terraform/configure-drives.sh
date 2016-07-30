#!/usr/bin/env bash

# r3.8xlarge instance have to SSDs attached but not configured for use. This script configures
# them so that Spark and other applications can use fast SSD storage instead of EBS. /ssd-b gets
# used by Spark directly and /ssd-c is left for swap space and other uses such cloning code.

# NOTE: these drives need to be re-attached on reboot, see here for how:
# http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html

# Create a file system on ephemeral SSDs
sudo mkfs -t ext4 /dev/xvdb
sudo mkfs -t ext4 /dev/xvdc

# Create mount location
sudo mkdir /ssd-b
sudo mkdir /ssd-c

# Mount the drives to the file system
sudo mount /dev/xvdb /ssd-b
sudo mount /dev/xvdc /ssd-c

sudo chmod 777 /ssd-b
sudo chmod 777 /ssd-c
sudo chown ubuntu /ssd-b
sudo chown ubuntu /ssd-c
