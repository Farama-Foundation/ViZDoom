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

PACKAGE_DESTINATION_DIRECTORY="./bin/python${PYTHON_VERSION}"
PACKAGE_DESTINATION_PATH="${PACKAGE_DESTINATION_DIRECTORY}/pip_package"
PACKAGE_SOURCE="./src/lib_python/pip_package"
VIZDOOM_PATH="./bin/vizdoom"
PK3_PATH="./bin/vizdoom.pk3"
PYTHON_BIN_PATH="${PACKAGE_DESTINATION_DIRECTORY}/vizdoom.so"
FREEDOOM_PATH="./freedoom2.wad"
SCENARIOS_DEST_DIR="${PACKAGE_DESTINATION_PATH}/scenarios"
SCENARIOS_PATH="./scenarios"

rm -rf ${PACKAGE_DESTINATION_PATH}
cp -r ${PACKAGE_SOURCE} ${PACKAGE_DESTINATION_PATH}
cp ${VIZDOOM_PATH} ${PACKAGE_DESTINATION_PATH}
mv ${PYTHON_BIN_PATH} ${PACKAGE_DESTINATION_PATH}
cp ${FREEDOOM_PATH} ${PACKAGE_DESTINATION_PATH}
cp ${PK3_PATH} ${PACKAGE_DESTINATION_PATH}
mkdir -p ${SCENARIOS_DEST_DIR}
cp ${SCENARIOS_PATH}/*.wad  ${SCENARIOS_DEST_DIR}
cp ${SCENARIOS_PATH}/*.cfg  ${SCENARIOS_DEST_DIR}
mv ${SCENARIOS_DEST_DIR}/bots.cfg ${PACKAGE_DESTINATION_PATH}