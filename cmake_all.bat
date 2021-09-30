@echo off
setlocal enabledelayedexpansion

:: Configuration 
:: %%% Set this variables to match your environment.
:: %%% CMake for Windws can be downloaded from https://cmake.org/download/

:: %%% Set build config
set PYTHON=ON

:: %%% Set your Visual Studio version
set CMAKE_GENERATOR_NAME=Visual Studio 16 2019

:: %%% Set path to folder with libraries and clone/copy https://github.com/mwydmuch/ViZDoomWinDepBin to it.
set LIB_DIR=C:\libs

:: API dependencies
:: %%% Set path to Boost library
:: %%% Prebuild Boost for MSVC can be downloaded from https://sourceforge.net/projects/boost/files/boost-binaries/
set BOOST_ROOT=%LIB_DIR%\boost
set BOOST_INCLUDEDIR=%BOOST_ROOT%
set BOOST_LIBRARYDIR=%BOOST_ROOT%\libs

:: Python
:: %%% Set Python version (35, 36, 37, 38, 39) or change paths for other distributions
:: %%% Python for Windows can be downloaded from https://www.python.org/downloads/windows/
set PYTHON_LOCATION=C:
set PYTHON_VERSION=39
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

:: ZDoom dependencies
set MPG123_INCLUDE_DIR=%LIB_DIR%\libmpg123
set MPG123_LIBRARIES=%LIB_DIR%\libmpg123\libmpg123-0.lib
set SNDFILE_INCLUDE_DIR=%LIB_DIR%\libsndfile\include
set SNDFILE_LIBRARY=%LIB_DIR%\libsndfile\lib\libsndfile-1.lib
:: set OPENAL_INCLUDE_DIR=%LIB_DIR%\openal-soft\include\AL
:: set OPENAL_LIBRARY=%LIB_DIR%\openal-soft\libs\Win64\OpenAL32.lib
set OPENALDIR=%LIB_DIR%\openal-soft

:: CMake command
:: %%% Minimal version
set CMAKE_CMD=-G "%CMAKE_GENERATOR_NAME%" -DCMAKE_BUILD_TYPE=Release -DBOOST_INCLUDEDIR="%BOOST_INCLUDEDIR%" -DBOOST_LIBRARYDIR="%BOOST_LIBRARYDIR%" -DBOOST_ROOT="%BOOST_ROOT%" -DNO_ASM=ON

:: %%% Version with all additional sound deps
:: set CMAKE_CMD=-G "%CMAKE_GENERATOR_NAME%" -DCMAKE_BUILD_TYPE=Release -DBOOST_INCLUDEDIR="%BOOST_INCLUDEDIR%" -DBOOST_LIBRARYDIR="%BOOST_LIBRARYDIR%" -DBOOST_ROOT="%BOOST_ROOT%" -DMPG123_INCLUDE_DIR="%MPG123_INCLUDE_DIR%" -DMPG123_LIBRARIES="%MPG123_LIBRARIES%" -DSNDFILE_INCLUDE_DIR="%SNDFILE_INCLUDE_DIR%" -DOPENAL_LIBRARY="%OPENAL_LIBRARY%" -DNO_ASM=ON -DFMOD_INCLUDE_DIR="%FMOD_INCLUDE_DIR%" -DFMOD_LIBRARY="%FMOD_LIBRARY%" 

if "%PYTHON%"=="ON" (     
    set CMAKE_CMD=!CMAKE_CMD! -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE="%PYTHON_EXECUTABLE%" -DPYTHON_INCLUDE_DIR="%PYTHON_INCLUDE_DIR%" -DPYTHON_LIBRARY="%PYTHON_LIBRARY%" -DNUMPY_INCLUDES=%NUMPY_INCLUDES%
)

del CMakeCache.txt
echo cmake !CMAKE_CMD!
cmake !CMAKE_CMD!
