@echo off
setlocal enabledelayedexpansion

:: Set to current ViZDoom version
set VIZDOOM_VERSION=1.2.1

:: Build wheels for all Python versions
for %%P in (38 39 310 311) do (
	set PYTHON_VERSION=%%P
	set PYTHON_VERSION_DOT=!PYTHON_VERSION:~0,1!.!PYTHON_VERSION:~1!
	echo Testing wheels for Python !PYTHON_VERSION_DOT! ...

	:: Modify these lines to point to your Python location (C:\PythonX is usually a default)
    set PYTHON_LOCATION=C:\Python!PYTHON_VERSION!
	set PYTHON_EXECUTABLE=!PYTHON_LOCATION!\python.exe

	!PYTHON_EXECUTABLE! -m pip uninstall -y vizdoom

	!PYTHON_EXECUTABLE! -m pip install --upgrade pip
	!PYTHON_EXECUTABLE! -m pip install scipy opencv-python pytest

	set WHEEL_FILE=dist\vizdoom-!VIZDOOM_VERSION!-cp!PYTHON_VERSION!-cp!PYTHON_VERSION!-win_amd64.whl
	if exist !WHEEL_FILE! (
		!PYTHON_EXECUTABLE! -m pip install !WHEEL_FILE![gym]
	)

	:: Test wheel from test PyPI index
	rem !PYTHON_EXECUTABLE! -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ vizdoom

	:: Run some examples
	!PYTHON_EXECUTABLE! examples\python\basic.py
	!PYTHON_EXECUTABLE! examples\python\buffers.py
	!PYTHON_EXECUTABLE! examples\python\audio_buffer.py

	:: Run tests
	!PYTHON_EXECUTABLE! -m pytest tests

	!PYTHON_EXECUTABLE! -m pip uninstall -y vizdoom
)
