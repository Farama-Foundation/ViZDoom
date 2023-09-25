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
python --version

# Install
export VIZDOOM_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/root/miniconda/"
python -m pip install .[test]

# Test import
python -c "import vizdoom"

# Run tests
pytest tests
