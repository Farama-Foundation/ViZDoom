#!/usr/bin/env bash

PYTHON_VERSION=$1
BIN_PATH=$2
SRC_PATH=$3

if [ $# -ne 3 ];then
    echo "Exactly three arguments required. Aborting."
    exit 1
fi

PACKAGE_DEST_DIRECTORY="${BIN_PATH}/python${PYTHON_VERSION}"
PACKAGE_DEST_PATH="${PACKAGE_DEST_DIRECTORY}/pip_package"
PACKAGE_INIT_FILE_SRC="${SRC_PATH}/src/lib_python/__init__.py"

if [ "$(uname)" == "Darwin" ]; then
    VIZDOOM_EXEC_PATH="${BIN_PATH}/vizdoom.app/Contents/MacOS/vizdoom"
else
    VIZDOOM_EXEC_PATH="${BIN_PATH}/vizdoom"
fi

VIZDOOM_PK3_PATH="${BIN_PATH}/vizdoom.pk3"
PYTHON_BIN_PATH="$(ls ${PACKAGE_DEST_DIRECTORY}/vizdoom*)"

FREEDOOM_PATH="${SRC_PATH}/src/freedoom2.wad"
SCENARIOS_DEST_DIR="${PACKAGE_DEST_PATH}/scenarios"
SCENARIOS_PATH="${SRC_PATH}/scenarios"

if [ ! -e ${PYTHON_BIN_PATH} ]; then
    echo "Library for specified Python version does not exist. Aborting."
    exit 2
fi

if [ ! -e ${VIZDOOM_EXEC_PATH} ] || [ ! -e ${VIZDOOM_PK3_PATH} ]; then
    echo "Required ViZDoom's resources do not exist. Aborting."
    exit 3
fi

rm -rf ${PACKAGE_DEST_PATH}
mkdir -p ${PACKAGE_DEST_PATH}

cp ${PACKAGE_INIT_FILE_SRC} ${PACKAGE_DEST_PATH}
cp ${PYTHON_BIN_PATH} ${PACKAGE_DEST_PATH}
cp ${VIZDOOM_EXEC_PATH} ${PACKAGE_DEST_PATH}
cp ${VIZDOOM_PK3_PATH} ${PACKAGE_DEST_PATH}
cp ${FREEDOOM_PATH} ${PACKAGE_DEST_PATH}
mkdir -p ${SCENARIOS_DEST_DIR}
cp ${SCENARIOS_PATH}/*.wad ${SCENARIOS_DEST_DIR}
cp ${SCENARIOS_PATH}/*.cfg ${SCENARIOS_DEST_DIR}
mv ${SCENARIOS_DEST_DIR}/bots.cfg ${PACKAGE_DEST_PATH}
