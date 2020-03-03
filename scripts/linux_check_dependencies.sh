#!/bin/bash

# NOTE: Tested on Ubuntu 16.04.

# SPECIFY USERNAME
UNAME=$1
CWD=pwd

# Install ZDoom dependencies
sudo apt-get install g++ make cmake libsdl2-dev git zlib1g-dev libbz2-dev \
libjpeg-dev libfluidsynth-dev libgme-dev libopenal-dev libmpg123-dev \
libsndfile1-dev libwildmidi-dev libgtk-3-dev timidity nasm tar chrpath

sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip

# Boost libraries
sudo apt-get install libboost-all-dev

# Python 3 dependencies
sudo apt-get install python3-dev python3-pip
pip3 install numpy

# Julia dependencies
sudo apt update
sudo apt -y install build-essential
cd /home/$UNAME
wget https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.1-linux-x86_64.tar.gz
tar xvfz julia-1.0.1-linux-x86_64.tar.gz
sudo ln -s /home/$UNAME/julia-1.0.1/bin/julia /usr/local/bin/julia

# Remove Gzipped Julia
rm -rf julia-1.0.1-linux-x86_64.tar.gz

# Source our bash
source ~/.bashrc

# Start julia
julia $CWD/set_up_cxxwrap.jl

