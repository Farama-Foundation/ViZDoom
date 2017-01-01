#!/usr/bin/env bash

PYTHON_VERSION=$1

if [ $# -ne 1 ];then
    echo "Exactly one argument required: python version = {2,3}. Aborting."
    exit 1
fi
if [ $1 -ne 2 ] && [ $1 -ne 3 ];then
    echo "Python version should be '2' or '3'. Aborting."
    exit 2
fi

PACKAGE_DEST_DIRECTORY="./bin/python${PYTHON_VERSION}"
PACKAGE_DEST_PATH="${PACKAGE_DEST_DIRECTORY}/pip_package"
PACKAGE_SOURCE="./src/lib_python/src_python"
if [ "$(uname)" == "Darwin" ]
then
    VIZDOOM_EXEC_PATH="./bin/vizdoom.app/Contents/MacOS/vizdoom"
else
    VIZDOOM_EXEC_PATH="./bin/vizdoom"
fi

PK3_PATH="./bin/vizdoom.pk3"
PYTHON_BIN_PATH="${PACKAGE_DEST_DIRECTORY}/vizdoom.so"
FREEDOOM_PATH="./freedoom2.wad"
SCENARIOS_DEST_DIR="${PACKAGE_DEST_PATH}/scenarios"
SCENARIOS_PATH="./scenarios"

rm -rf ${PACKAGE_DEST_PATH}
cp -r ${PACKAGE_SOURCE} ${PACKAGE_DEST_PATH}
cp ${VIZDOOM_EXEC_PATH} ${PACKAGE_DEST_PATH}
cp ${PYTHON_BIN_PATH} ${PACKAGE_DEST_PATH}
cp ${FREEDOOM_PATH} ${PACKAGE_DEST_PATH}
cp ${PK3_PATH} ${PACKAGE_DEST_PATH}
mkdir -p ${SCENARIOS_DEST_DIR}
cp ${SCENARIOS_PATH}/*.wad  ${SCENARIOS_DEST_DIR}
cp ${SCENARIOS_PATH}/*.cfg  ${SCENARIOS_DEST_DIR}
mv ${SCENARIOS_DEST_DIR}/bots.cfg ${PACKAGE_DEST_PATH}