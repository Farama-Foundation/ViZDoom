@echo off
setlocal enabledelayedexpansion

:: Configuration 
:: %%% Set this variables to match your environment.
:: %%% CMake for Windws can be downloaded from https://cmake.org/download/

:: %%% Set build config
set PYTHON=ON
set LUA=OFF
set JAVA=OFF
set CMAKE_GENERATOR_NAME=Visual Studio 15 2017 Win64

:: %%% Set path to folder with libraries and clone/copy https://github.com/mwydmuch/ViZDoomWinDepBin to it.
set LIB_DIR=C:\libs

:: API dependencies
:: %%% Set path to Boost library
:: %%% Prebuild Boost for MSVC can be downloaded from https://sourceforge.net/projects/boost/files/boost-binaries/
set BOOST_ROOT=%LIB_DIR%\boost
set BOOST_INCLUDEDIR=%BOOST_ROOT%
set BOOST_LIBRARYDIR=%BOOST_ROOT%\libs

:: Python
:: %%% Set Python version (27, 35, 36, 37) or change paths for other distributions
:: %%% Python for Windows can be downloaded from https://www.python.org/downloads/windows/
set PYTHON_LOCATION=C:
set PYTHON_VERSION=37
set PYTHON_BIG_VERSION=%PYTHON_VERSION:~0,1%
set PYTHON_EXECUTABLE=%PYTHON_LOCATION%\Python%PYTHON_VERSION%\python.exe
set PYTHON_INCLUDE_DIR=%PYTHON_LOCATION%\Python%PYTHON_VERSION%\include
set PYTHON_LIBRARY=%PYTHON_LOCATION%\Python%PYTHON_VERSION%\libs\python%PYTHON_VERSION%.lib
set NUMPY_INCLUDES=%PYTHON_LOCATION%\Python%PYTHON_VERSION%\Lib\site-packages\numpy\core\include

:: %%% Install/upgrade pip & numpy
%PYTHON_LOCATION%\Python%PYTHON_VERSION%\python.exe -m pip install --upgrade pip
%PYTHON_LOCATION%\Python%PYTHON_VERSION%\python.exe -m pip install --upgrade numpy

:: TODO: Add default Anaconda paths
:: TODO: Add Julia support

:: Rest of the script
::--------------------------------------------------------------------------------------------------------------------

:: Lua
set LUA_INCLUDE_DIR=%LIB_DIR%\lua\include
set LUA_LIBRARY=%LIB_DIR%\lua\lua5.1.lib

:: ZDoom dependencies
set FMOD_INCLUDE_DIR=%LIB_DIR%\fmod\inc
set FMOD_LIBRARY=%LIB_DIR%\fmod\lib\fmodex64.lib
set MPG123_INCLUDE_DIR=%LIB_DIR%\libmpg123
set MPG123_LIBRARIES=%LIB_DIR%\libmpg123\libmpg123-0.lib
set SNDFILE_INCLUDE_DIR=%LIB_DIR%\libsndfile\include
set SNDFILE_LIBRARY=%LIB_DIR%\libsndfile\lib\libsndfile-1.lib
set OPENAL_INCLUDE_DIR=%LIB_DIR%\openal\include
set OPENAL_LIBRARY=%LIB_DIR%\openal\libs\openal32.lib
set YASM_PATH=%LIB_DIR%\yasm.exe

:: CMake command
set CMAKE_CMD=-G "%CMAKE_GENERATOR_NAME%" -DCMAKE_BUILD_TYPE=Release -DBOOST_INCLUDEDIR="%BOOST_INCLUDEDIR%" -DBOOST_LIBRARYDIR="%BOOST_LIBRARYDIR%" -DBOOST_ROOT="%BOOST_ROOT%" -DFMOD_INCLUDE_DIR="%FMOD_INCLUDE_DIR%"  -DFMOD_LIBRARY="%FMOD_LIBRARY%" -DMPG123_INCLUDE_DIR="%MPG123_INCLUDE_DIR%" -DMPG123_LIBRARIES="%MPG123_LIBRARIES%" -DSNDFILE_INCLUDE_DIR="%SNDFILE_INCLUDE_DIR%" -DSNDFILE_LIBRARY="%SNDFILE_LIBRARY%" -DOPENAL_INCLUDE_DIR="%OPENAL_INCLUDE_DIR%" -DOPENAL_LIBRARY="%OPENAL_LIBRARY%" -DNO_ASM=ON 
:: -DYASM_PATH="%YASM_PATH%"

if "%LUA%"=="ON" (
    set CMAKE_CMD=!CMAKE_CMD! -DBUILD_LUA=ON -DLUA_INCLUDE_DIR="%LUA_INCLUDE_DIR%" -DLUA_LIBRARY="%LUA_LIBRARY%"
)

if "%JAVA%"=="ON" (
    set CMAKE_CMD=!CMAKE_CMD! -DBUILD_JAVA=ON
)

if "%PYTHON%"=="ON" ( 
    if %PYTHON_BIG_VERSION% equ 2 (
        set CMAKE_CMD=!CMAKE_CMD! -DBUILD_PYTHON=ON
    )
    
    if %PYTHON_BIG_VERSION% equ 3 (
        set CMAKE_CMD=!CMAKE_CMD! -DBUILD_PYTHON3=ON
    )
    
    set CMAKE_CMD=!CMAKE_CMD! -DPYTHON_EXECUTABLE="%PYTHON_EXECUTABLE%" -DPYTHON_INCLUDE_DIR="%PYTHON_INCLUDE_DIR%" -DPYTHON_LIBRARY="%PYTHON_LIBRARY%" -DNUMPY_INCLUDES=%NUMPY_INCLUDES%
)

del CMakeCache.txt
echo cmake !CMAKE_CMD!
cmake !CMAKE_CMD!

