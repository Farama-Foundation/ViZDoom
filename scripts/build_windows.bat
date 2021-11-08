@echo off
setlocal enabledelayedexpansion

:: Configuration 
:: %%% Set this variables to match your environment.
:: %%% CMake for Windws can be downloaded from https://cmake.org/download/

:: %%% Set your Visual Studio version
set CMAKE_GENERATOR_NAME=Visual Studio 16 2019

:: %%% Set path to folder with libraries and clone/copy https://github.com/mwydmuch/ViZDoomWinDepBin to it.
set LIB_DIR=C:\ViZDoomWinDepBin

:: API dependencies
:: %%% Set path to Boost library
:: %%% Prebuild Boost for MSVC can be downloaded from https://sourceforge.net/projects/boost/files/boost-binaries/
set BOOST_ROOT=%LIB_DIR%\boost
set BOOST_INCLUDEDIR=%BOOST_ROOT%
set BOOST_LIBRARYDIR=%BOOST_ROOT%\libs

:: ZDoom dependencies
set MPG123_INCLUDE_DIR=%LIB_DIR%\libmpg123
set MPG123_LIBRARIES=%LIB_DIR%\libmpg123\libmpg123-0.lib
set MPG123_DLL=%LIB_DIR%\libmpg123\libmpg123-0.dll
set SNDFILE_INCLUDE_DIR=%LIB_DIR%\libsndfile\include
set SNDFILE_LIBRARY=%LIB_DIR%\libsndfile\lib\libsndfile-1.lib
set SNDFILE_DLL=%LIB_DIR%\libsndfile\bin\libsndfile-1.dll
set OPENALDIR=%LIB_DIR%\openal-soft
set OPENAL_DLL=%LIB_DIR%\openal-soft\bin\Win64\OpenAL32.dll

:: Build wheels for all Python versions
for %%P in (36 37 38 39 310) do (
	set PYTHON_VERSION=%%P
	set PYTHON_VERSION_DOT=!PYTHON_VERSION:~0,1!.!PYTHON_VERSION:~1!
	echo Building for Python !PYTHON_VERSION_DOT! version using !CMAKE_GENERATOR_NAME!
	
	set PYTHON_LOCATION=C:\Python!PYTHON_VERSION!
	set PYTHON_EXECUTABLE=!PYTHON_LOCATION!\python.exe
	set PYTHON_INCLUDE_DIR=!PYTHON_LOCATION!\include
	set PYTHON_LIBRARY=!PYTHON_LOCATION!\libs\python!PYTHON_VERSION!.lib
	
	!PYTHON_EXECUTABLE! -m pip install --upgrade pip
    !PYTHON_EXECUTABLE! -m pip install --upgrade numpy

	:: CMake command
	:: %%% Minimal version
	:: set CMAKE_CMD=-G "%CMAKE_GENERATOR_NAME%" -DCMAKE_BUILD_TYPE=Release -DBOOST_INCLUDEDIR="%BOOST_INCLUDEDIR%" -DBOOST_LIBRARYDIR="%BOOST_LIBRARYDIR%" -DBOOST_ROOT="%BOOST_ROOT%" -DNO_ASM=ON
	:: %%% Version with all additional sound deps
	set CMAKE_CMD=-G "%CMAKE_GENERATOR_NAME%" -DCMAKE_BUILD_TYPE=Release -DBOOST_INCLUDEDIR="%BOOST_INCLUDEDIR%" -DBOOST_LIBRARYDIR="%BOOST_LIBRARYDIR%" -DBOOST_ROOT="%BOOST_ROOT%" -DNO_ASM=ON -DMPG123_INCLUDE_DIR="%MPG123_INCLUDE_DIR%" -DMPG123_LIBRARIES="%MPG123_LIBRARIES%" -DSNDFILE_INCLUDE_DIR="%SNDFILE_INCLUDE_DIR%" -DSNDFILE_LIBRARY=%SNDFILE_LIBRARY%

	set CMAKE_CMD=!CMAKE_CMD! -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE="!PYTHON_EXECUTABLE!" -DPYTHON_INCLUDE_DIR="!PYTHON_INCLUDE_DIR!" -DPYTHON_LIBRARY="!PYTHON_LIBRARY!"

	del CMakeCache.txt
	rd /S /Q .\src\lib_python\libvizdoom_python.dir
	echo cmake !CMAKE_CMD!

	cmake !CMAKE_CMD!
	
	:: Run build
	cmake --build . --config Release

	copy /Y !MPG123_DLL! .\bin\python!PYTHON_VERSION_DOT!\pip_package\
	copy /Y !SNDFILE_DLL! .\bin\python!PYTHON_VERSION_DOT!\pip_package\
	copy /Y !OPENAL_DLL! .\bin\python!PYTHON_VERSION_DOT!\pip_package\
)
