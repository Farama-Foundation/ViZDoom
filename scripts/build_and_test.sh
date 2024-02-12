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
python3 --version
python3 -m pip --version

# Workaround for environment being externally managed in system installed Python 3.11+\
PYTHON_VERSION_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
PIP_VERSION_MAJOR=$(python3 -c "import pip; print(pip.__version__.split('.')[0])")
PIP_FLAGS=""
if (( ${PYTHON_VERSION_MINOR} >= 11 && ${PIP_VERSION_MAJOR} >= 23)); then
    echo "Adding --break-system-packages flag to pip install"
    PIP_FLAGS="--break-system-packages"
fi

# Updgrad pip and install test deps
python3 -m pip install --upgrade pip ${PIP_FLAGS}
python3 -m pip install .[test] ${PIP_FLAGS}

# Test import
python3 -c "import vizdoom"

# Run tests
pytest tests
