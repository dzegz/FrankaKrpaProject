#!/bin/bash
# Project: AK_ACQS Azure Kinect Acquisition System https://github.com/GRAP-UdL-AT/ak_acquisition_system
#
# * PAgFRUIT http://www.pagfruit.udl.cat/en/
# * GRAP http://www.grap.udl.cat/
#
# Author: Juan Carlos Miranda. https://github.com/juancarlosmiranda
# Modified for use in Docker by Matej.

# Install repositories Ubuntu 20.04 and Ubuntu 18.04
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc
apt-add-repository https://packages.microsoft.com/ubuntu/20.04/prod
apt-get update

curl -sSL https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
curl -sSL https://packages.microsoft.com/config/ubuntu/18.04/prod.list | tee /etc/apt/sources.list.d/microsoft-prod.list
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
apt-get update

#Install new package from repository
apt-get update
ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
    libk4a1.4 \
    libk4a1.4-dev \
    k4a-tools
