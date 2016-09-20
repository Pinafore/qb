#!/usr/bin/env bash

set -e

# Install Pandoc
cd ~/dependencies
wget https://github.com/jgm/pandoc/releases/download/1.17.2/pandoc-1.17.2-1-amd64.deb
ar p pandoc-1.17.2-1-amd64.deb data.tar.gz | sudo tar xvz --strip-components 2 -C /usr/local
