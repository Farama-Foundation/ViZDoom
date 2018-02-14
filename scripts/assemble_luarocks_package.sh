#!/usr/bin/env bash

PACKAGE_DEST_DIRECTORY="./bin/lua"
PACKAGE_DEST_PATH="${PACKAGE_DEST_DIRECTORY}/luarocks_package"
SHARED_PACKAGE_DEST_PATH="${PACKAGE_DEST_DIRECTORY}/luarocks_shared_package"
PACKAGE_SOURCE="./src/lib_lua/src_lua"
if [ "$(uname)" == "Darwin" ]; then
    VIZDOOM_EXEC_PATH="./bin/vizdoom.app/Contents/MacOS/vizdoom"
    LUA_BIN_PATH="${PACKAGE_DEST_DIRECTORY}/vizdoom.dylib"
else
    VIZDOOM_EXEC_PATH="./bin/vizdoom"
    LUA_BIN_PATH="${PACKAGE_DEST_DIRECTORY}/vizdoom.so"
fi

VIZDOOM_PK3_PATH="./bin/vizdoom.pk3"
FREEDOOM_PATH="./bin/freedoom2.wad"
SCENARIOS_DEST_DIR="${PACKAGE_DEST_PATH}/scenarios"
SCENARIOS_PATH="./scenarios"

if [ ! -e ${LUA_BIN_PATH} ]; then
    echo "Library for Lua does not exist. Aborting."
    exit 1
fi

if [ ! -e ${VIZDOOM_EXEC_PATH} ] || [ ! -e ${VIZDOOM_PK3_PATH} ]; then
    echo "Required ViZDoom's resources do not exist. Aborting."
    exit 2
fi

rm -rf ${PACKAGE_DEST_PATH}
rm -rf ${SHARED_PACKAGE_DEST_PATH}
mkdir -p ${PACKAGE_DEST_PATH}
mkdir -p ${SHARED_PACKAGE_DEST_PATH}

cp -r ${PACKAGE_SOURCE}/* ${SHARED_PACKAGE_DEST_PATH}
cp -r ${PACKAGE_SOURCE}/* ${PACKAGE_DEST_PATH}
cp ${LUA_BIN_PATH} ${PACKAGE_DEST_PATH}
cp ${VIZDOOM_EXEC_PATH} ${PACKAGE_DEST_PATH}
cp ${VIZDOOM_PK3_PATH} ${PACKAGE_DEST_PATH}
cp ${FREEDOOM_PATH} ${PACKAGE_DEST_PATH}
mkdir -p ${SCENARIOS_DEST_DIR}
cp ${SCENARIOS_PATH}/*.wad  ${SCENARIOS_DEST_DIR}
cp ${SCENARIOS_PATH}/*.cfg  ${SCENARIOS_DEST_DIR}
mv ${SCENARIOS_DEST_DIR}/bots.cfg ${PACKAGE_DEST_PATH}
