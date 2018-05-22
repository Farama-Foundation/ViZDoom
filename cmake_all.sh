#!/bin/bash

# Set DBUILD options to match your needs.
rm CMakeCache.txt
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=OFF -DBUILD_PYTHON3=ON -DBUILD_JAVA=OFF -DBUILD_LUA=OFF
