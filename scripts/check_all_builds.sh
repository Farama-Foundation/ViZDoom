#!/usr/bin/env bash

# Right now this script is designed only for Ubuntu
# TODO: add MacOS support
# TODO: add support for different boost versions

set -e

# Start in repo's root dir
CURRENT_DIR=$( pwd )
SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )
cd ${SCRIPT_DIR}/..

# All combinations of GCCS and PYTHONS will be checked
GCCS=(6 7 8)
PYTHONS=(3.5 3.6 3.7)

for gcc in ${GCCS[*]}; do
    for python in ${PYTHONS[*]}; do
        if [ ! -f Makefile ]; then
            make clean
        fi
        rm -f CMakeCache.txt
        logfile=${CURRENT_DIR}/build_log_gcc${gcc}_python${python}.log

        echo "Checking build with gcc-${gcc} and python${python}, log file: ${logfile}"

        cmake -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER=/usr/bin/gcc-${gcc} -DCMAKE_CXX_COMPILER=/usr/bin/g++-${gcc} \
            -DBUILD_PYTHON3=ON \
            -DPYTHON_INCLUDE_DIR=/usr/include/python${python} \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython${python}m.so \
            -DPYTHON_EXECUTABLE=/usr/bin/python${python} \
            -DNUMPY_INCLUDES=${HOME}/.local/lib/python${python}/site-packages/numpy/core/includeinclude &> ${logfile}
        make -j &>> ${logfile}
        rm ${logfile}
    done
done
