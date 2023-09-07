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

# Install
python3 -m pip install .[test]

# Test import
python3 -c "import vizdoom"

# Run tests
pytest tests
