#!/usr/bin/env bash
set -e

# Set working dir to the root of the repo
cd $( dirname "${BASH_SOURCE[0]}" )/..

# Report directory
ls -lha .

# Report cmake version
cmake --version

# Report gcc version
gcc --version

# Report python version
python3 -c "import sys; print('Python', sys.version)"

# # Install
# python3 -m pip install .

# # Test import
# python3 -c "import vizdoom"

# # # Install rest deps
# # python3 -m pip install pytest

# Run tests
echo "Conda prefix: ${CONDA_PREFIX}"

# CMake manual build
rm CMakeCache.txt
cmake -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON .
make -j
