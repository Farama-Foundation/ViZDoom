@echo off
setlocal enabledelayedexpansion

:: Build wheels for all Python versions
for %%P in (36 37 38 39 310) do (
	set PYTHON_VERSION=%%P
	set PYTHON_VERSION_DOT=!PYTHON_VERSION:~0,1!.!PYTHON_VERSION:~1!
	echo Testing wheels for Python !PYTHON_VERSION_DOT! ...
	
    set PYTHON_LOCATION=E:\Python!PYTHON_VERSION!
	set PYTHON_EXECUTABLE=!PYTHON_LOCATION!\python.exe
	
	!PYTHON_EXECUTABLE! -m pip install dist\vizdoom-1.1.10-cp!PYTHON_VERSION!-cp!PYTHON_VERSION!-win_amd64.whl
	!PYTHON_EXECUTABLE! -m pip install dist\vizdoom-1.1.10-cp!PYTHON_VERSION!-cp!PYTHON_VERSION!m-win_amd64.whl
	
	:: ### !PYTHON_EXECUTABLE! examples\python\basic.py
	!PYTHON_EXECUTABLE! tests\test_enums.py
	!PYTHON_EXECUTABLE! tests\test_make_action.py
	!PYTHON_EXECUTABLE! tests\test_get_state.py
	
	!PYTHON_EXECUTABLE! -m pip uninstall -y vizdoom
)
