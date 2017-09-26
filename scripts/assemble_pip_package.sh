#!/usr/bin/env bash

PYTHON_VERSION=$1

if [ $# -ne 1 ];then
    echo "Exactly one argument required. Aborting."
    exit 1
fi

PACKAGE_DEST_DIRECTORY="./bin/python${PYTHON_VERSION}"
PACKAGE_DEST_PATH="${PACKAGE_DEST_DIRECTORY}/pip_package"
PACKAGE_SOURCE="./src/lib_python/src_python"
if [ "$(uname)" == "Darwin" ]; then
    VIZDOOM_EXEC_PATH="./bin/vizdoom.app/Contents/MacOS/vizdoom"
else
    VIZDOOM_EXEC_PATH="./bin/vizdoom"
fi

VIZDOOM_PK3_PATH="./bin/vizdoom.pk3"
PYTHON_BIN_PATH="$(ls ${PACKAGE_DEST_DIRECTORY}/vizdoom*.so)"

FREEDOOM_PATH="./bin/freedoom2.wad"
SCENARIOS_DEST_DIR="${PACKAGE_DEST_PATH}/scenarios"
SCENARIOS_PATH="./scenarios"

if [ ! -e ${PYTHON_BIN_PATH} ]; then
    echo "Library for specified Python version does not exist. Aborting."
    exit 2
fi

if [ ! -e ${VIZDOOM_EXEC_PATH} ] || [ ! -e ${VIZDOOM_PK3_PATH} ]; then
    echo "Required ViZDoom's resources do not exist. Aborting."
    exit 3
fi

rm -rf ${PACKAGE_DEST_PATH}
cp -r ${PACKAGE_SOURCE} ${PACKAGE_DEST_PATH}
cp ${VIZDOOM_EXEC_PATH} ${PACKAGE_DEST_PATH}
cp ${PYTHON_BIN_PATH} ${PACKAGE_DEST_PATH}
mv ${PYTHON_BIN_PATH} "${PACKAGE_DEST_PATH}/vizdoom.so"
cp ${FREEDOOM_PATH} ${PACKAGE_DEST_PATH}
cp ${VIZDOOM_PK3_PATH} ${PACKAGE_DEST_PATH}
mkdir -p ${SCENARIOS_DEST_DIR}
cp ${SCENARIOS_PATH}/*.wad  ${SCENARIOS_DEST_DIR}
cp ${SCENARIOS_PATH}/*.cfg  ${SCENARIOS_DEST_DIR}
mv ${SCENARIOS_DEST_DIR}/bots.cfg ${PACKAGE_DEST_PATH}
