@echo off
setlocal enabledelayedexpansion

:: Configuration 
:: %%% Set this variables to match your environment.
:: %%% CMake for Windws can be downloaded from https://cmake.org/download/

:: %%% Set your Visual Studio version
set VIZDOOM_BUILD_GENERATOR_NAME=Visual Studio 16 2019

:: %%% Set path to folder with libraries and clone/copy https://github.com/mwydmuch/ViZDoomWinDepBin to it.
set VIZDOOM_WIN_DEPS_ROOT=C:/ViZDoomWinDepBin

:: API dependencies
:: %%% Set path to Boost library
:: %%% Prebuild Boost for MSVC can be downloaded from https://sourceforge.net/projects/boost/files/boost-binaries/
set BOOST_ROOT=%VIZDOOM_WIN_DEPS_ROOT%/boost


:: Build wheels for all Python versions
for %%P in (36 37 38 39 310) do (
	set PYTHON_VERSION=%%P
	set PYTHON_VERSION_DOT=!PYTHON_VERSION:~0,1!.!PYTHON_VERSION:~1!
	echo Building wheel for Python !PYTHON_VERSION_DOT! version using !VIZDOOM_BUILD_GENERATOR_NAME!
	
	set PYTHON_LOCATION=C:\Python!PYTHON_VERSION!
	set PYTHON_EXECUTABLE=!PYTHON_LOCATION!\python.exe
	
	!PYTHON_EXECUTABLE! -m pip install --upgrade pip
    !PYTHON_EXECUTABLE! -m pip install --upgrade numpy
	!PYTHON_EXECUTABLE! -m pip install --upgrade setuptools wheel twine
	!PYTHON_EXECUTABLE! setup.py bdist_wheel
)
