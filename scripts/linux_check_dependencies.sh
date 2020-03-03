#!/bin/bash

# NOTE: Tested on Ubuntu 16.04.

# Install ZDoom dependencies
sudo apt-get install g++ make cmake libsdl2-dev git zlib1g-dev libbz2-dev \
libjpeg-dev libfluidsynth-dev libgme-dev libopenal-dev libmpg123-dev \
libsndfile1-dev libwildmidi-dev libgtk2.0-dev libgtk-3-dev timidity nasm tar chrpath \
build-essential unzip

# Boost libraries
sudo apt-get install libboost-all-dev

# Python 3 dependencies
sudo apt-get install python3-dev python3-pip
pip3 install numpy


