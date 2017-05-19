#!/usr/bin/env bash

# r3.8xlarge instance have to SSDs attached but not configured for use. This script configures
# them so that Spark and other applications can use fast SSD storage instead of EBS. /ssd-b gets
# used by Spark directly and /ssd-c is left for swap space and other uses such cloning code.
# This script will use ephemeral drives mounted on xvdb/xvdc when available, if not it will assume
# that xvdd exists if xvdb doesn't and xvde exists if xvdc doesn't

# NOTE: these drives need to be re-attached on reboot, see here for how:
# http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html

# Create a file system on ephemeral SSDs
# Create mount location
# Mount the drives to the file system
if [ -b /dev/xvdb ]; then
    sudo mkfs -t ext4 /dev/xvdb
    sudo mkdir /ssd-b
    sudo mount /dev/xvdb /ssd-b
elif [ -b /dev/xvdd ]; then
    sudo mkfs -t ext4 /dev/xvdd
    sudo mkdir /ssd-b
    sudo mount /dev/xvdd /ssd-b
else
    echo "Could not create /ssd-b since /dev/xvdb and /dev/xvdd don't exist, exiting with error..."
    exit 1
fi

if [ -b /dev/xvdc ]; then
    sudo mkfs -t ext4 /dev/xvdc
    sudo mkdir /ssd-c
    sudo mount /dev/xvdc /ssd-c
elif [ -b /dev/xvde ]; then
    sudo mkfs -t ext4 /dev/xvde
    sudo mkdir /ssd-c
    sudo mount /dev/xvde /ssd-c
else
    echo "Could not create /ssd-c since /dev/xvdc and /dev/xvde don't exist, exiting with error..."
    exit 1
fi

sudo chmod 777 /ssd-b
sudo chmod 777 /ssd-c
sudo chown ubuntu /ssd-b
sudo chown ubuntu /ssd-c
