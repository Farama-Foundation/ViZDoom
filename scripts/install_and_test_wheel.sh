#!/usr/bin/env bash
set -e

# Set working dir to the root of the repo
cd $( dirname "${BASH_SOURCE[0]}" )/..

# Report directory
ls -lha .

# Report python version
python3 --version
python3 -c "import sys; print('Python', sys.version)"

# Find matching wheel file in wheelhouse
PYTHON_VERSION=$(python3 -c "import sys; print('{}{}'.format(sys.version_info.major, sys.version_info.minor))")
PYTHON_WHEEL=$(ls wheelhouse/vizdoom-*-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}*.whl)

# Updgrad pip and install test deps
python3 -m pip install --upgrade pip
python3 -m pip install pytest psutil

# Install wheel
python3 -m pip install ${PYTHON_WHEEL}

# Test import
python3 -c "import vizdoom"

# Run tests
pytest tests
