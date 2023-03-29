:: @echo off
setlocal enabledelayedexpansion

set PYTHON_VERSION=%1
set BIN_PATH=%2
set SRC_PATH=%3

:: Replace back-slashes with forward-slashes
set "BIN_PATH=%BIN_PATH:/=\%"
set "SRC_PATH=%SRC_PATH:/=\%"

set PACKAGE_DEST_DIRECTORY=%BIN_PATH%\python%PYTHON_VERSION%
set PACKAGE_DEST_PATH=%PACKAGE_DEST_DIRECTORY%\pip_package
set PACAKGE_INIT_FILE_SRC=%SRC_PATH%\src\lib_python\__init__.py

set VIZDOOM_EXEC_PATH=%BIN_PATH%\vizdoom.exe
set VIZDOOM_PK3_PATH=%BIN_PATH%\vizdoom.pk3
set FREEDOOM_PATH=%SRC_PATH%\src\freedoom2.wad
set BOTS_PATH=%SRC_PATH%\src\bots.cfg
set SCENARIOS_DEST_DIR=%PACKAGE_DEST_PATH%\scenarios
set SCENARIOS_PATH=%SRC_PATH%\scenarios
set GYM_WRAPPER_DEST_DIR=%PACKAGE_DEST_PATH%\gym_wrapper
set GYM_WRAPPER_PATH=%SRC_PATH%\gym_wrapper
set GYMNASIUM_WRAPPER_DEST_DIR=%PACKAGE_DEST_PATH%\gymnasium_wrapper
set GYMNASIUM_WRAPPER_PATH=%SRC_PATH%\gymnasium_wrapper


if not exist "%BIN_PATH%\python%PYTHON_VERSION%\vizdoom*.pyd" (
    echo "Library for specified Python version does not exist. Aborting."
    exit /B 2
)

if not exist "%VIZDOOM_EXEC_PATH%" (
    echo "Required ViZDoom's resources do not exist. Aborting."
    exit /B 3
)

if not exist "%VIZDOOM_PK3_PATH%" (
    echo "Required ViZDoom's resources do not exist. Aborting."
    exit /B 3
)

rmdir /Q /S %PACKAGE_DEST_PATH%
md %PACKAGE_DEST_PATH%

copy "%PACAKGE_INIT_FILE_SRC%" "%PACKAGE_DEST_PATH%"
copy "%VIZDOOM_EXEC_PATH%" "%PACKAGE_DEST_PATH%"
copy "%VIZDOOM_PK3_PATH%" "%PACKAGE_DEST_PATH%"
copy "%BIN_PATH%\python%PYTHON_VERSION%\vizdoom*.pyd" "%PACKAGE_DEST_PATH%"
copy "%FREEDOOM_PATH%" "%PACKAGE_DEST_PATH%"
copy "%BOTS_PATH%" "%PACKAGE_DEST_PATH%"
copy "%BIN_PATH%\*.dll" "%PACKAGE_DEST_PATH%"

md "%SCENARIOS_DEST_DIR%
copy "%SCENARIOS_PATH%\*.wad" "%SCENARIOS_DEST_DIR%"
copy "%SCENARIOS_PATH%\*.cfg" "%SCENARIOS_DEST_DIR%"

md "%GYM_WRAPPER_DEST_DIR%
copy "%GYM_WRAPPER_PATH%\*.py" "%GYM_WRAPPER_DEST_DIR%"
md "%GYMNASIUM_WRAPPER_DEST_DIR%
copy "%GYMNASIUM_WRAPPER_PATH%\*.py" "%GYMNASIUM_WRAPPER_DEST_DIR%"
