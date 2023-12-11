#!/usr/bin/env bash
set -e

# Set working dir to the root of the repo
cd $( dirname "${BASH_SOURCE[0]}" )/..

# Report directory
ls -lha .

# Report python version
python3 --version
python3 -m pip --version

# Find matching wheel file in wheelhouse
PYTHON_VERSION_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_VERSION_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
PYTHON_VERSION="${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}"
PYTHON_WHEEL=$(ls wheelhouse/vizdoom-*-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}*.whl)

if [ -z "${PYTHON_WHEEL}" ]; then
    echo "No matching wheel file was found"
    exit 1
fi

# Workaround for environment being externally managed in system installed Python 3.11+
PIP_VERSION_MAJOR=$(python3 -c "import pip; print(pip.__version__.split('.')[0])")
PIP_FLAGS=""
if (( ${PYTHON_VERSION_MINOR} >= 11 && ${PIP_VERSION_MAJOR} >= 23)); then
    echo "Adding --break-system-packages flag to pip install"
    PIP_FLAGS="--break-system-packages"
fi

# Updgrad pip and install test deps
python3 -m pip install --upgrade pip ${PIP_FLAGS}
python3 -m pip install pytest psutil ${PIP_FLAGS}

# Install wheel
python3 -m pip install ${PYTHON_WHEEL} ${PIP_FLAGS}

# Test import
python3 -c "import vizdoom"

# Run tests
pytest tests
