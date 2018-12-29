:: @echo off
setlocal enabledelayedexpansion

set PYTHON_VERSION=%1

set BIN_PATH=.\bin
set PACKAGE_DEST_DIRECTORY=%BIN_PATH%\python%PYTHON_VERSION%
set PACKAGE_DEST_PATH=%PACKAGE_DEST_DIRECTORY%\pip_package
set PACAKGE_INIT_FILE_SRC=.\src\lib_python\__init__.py

set VIZDOOM_EXEC_PATH=%BIN_PATH%\vizdoom.exe
set VIZDOOM_PK3_PATH=%BIN_PATH%\vizdoom.pk3
dir dir .\bin\python%PYTHON_VERSION%\vizdoom*.pyd /b /s > %PACKAGE_DEST_DIRECTORY%\tmp.txt
set /p PYTHON_BIN_PATH=<%PACKAGE_DEST_DIRECTORY%\tmp.txt
del %PACKAGE_DEST_DIRECTORY%\tmp.txt
set PYTHON_BIN_DEST_PATH=%PACKAGE_DEST_PATH%\vizdoom.pyd

set FREEDOOM_PATH=%BIN_PATH%\freedoom2.wad
set SCENARIOS_DEST_DIR=%PACKAGE_DEST_PATH%\scenarios
set SCENARIOS_PATH=.\scenarios
set EXAMPLES_DEST_DIR=%PACKAGE_DEST_PATH%\examples
set EXAMPLES_PATH=".\examples\python"

if not exist "%PYTHON_BIN_PATH%" (
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

del %PACKAGE_DEST_PATH%
md %PACKAGE_DEST_PATH%

copy "%PACAKGE_INIT_FILE_SRC%" "%PACKAGE_DEST_PATH%"
copy "%PYTHON_BIN_PATH%" "%PYTHON_BIN_DEST_PATH%"
copy "%VIZDOOM_EXEC_PATH%" "%PACKAGE_DEST_PATH%"
copy "%VIZDOOM_PK3_PATH%" "%PACKAGE_DEST_PATH%"
copy "%BIN_PATH%\*.pyd" "%PACKAGE_DEST_PATH%"
copy "%FREEDOOM_PATH%" "%PACKAGE_DEST_PATH%"
copy "%FREEDOOM_PATH%" "%PACKAGE_DEST_PATH%"
md "%SCENARIOS_DEST_DIR%
copy "%SCENARIOS_PATH%\*.wad" "%SCENARIOS_DEST_DIR%"
copy "%SCENARIOS_PATH%\*.cfg" "%SCENARIOS_DEST_DIR%"
md "%EXAMPLES_DEST_DIR%
copy "%EXAMPLES_PATH%\*.py" "%EXAMPLES_DEST_DIR%"
move "%SCENARIOS_DEST_DIR%\bots.cfg" "%PACKAGE_DEST_PATH%"
