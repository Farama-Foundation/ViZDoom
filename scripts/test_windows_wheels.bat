@echo off
setlocal enabledelayedexpansion

:: Build wheels for all Python versions
for %%P in (37 38 39 310) do (
	set PYTHON_VERSION=%%P
	set PYTHON_VERSION_DOT=!PYTHON_VERSION:~0,1!.!PYTHON_VERSION:~1!
	echo Testing wheels for Python !PYTHON_VERSION_DOT! ...
	
    set PYTHON_LOCATION=C:\Python!PYTHON_VERSION!
	set PYTHON_EXECUTABLE=!PYTHON_LOCATION!\python.exe
	
	!PYTHON_EXECUTABLE! -m pip uninstall -y vizdoom
	
	!PYTHON_EXECUTABLE! -m pip install --upgrade pip
	!PYTHON_EXECUTABLE! -m pip install scipy opencv-python pytest gym==0.23.0 pygame==2.1.0
	
	!PYTHON_EXECUTABLE! -m pip install dist\vizdoom-1.1.12-cp!PYTHON_VERSION!-cp!PYTHON_VERSION!-win_amd64.whl
	!PYTHON_EXECUTABLE! -m pip install dist\vizdoom-1.1.12-cp!PYTHON_VERSION!-cp!PYTHON_VERSION!m-win_amd64.whl
	rem !PYTHON_EXECUTABLE! -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ vizdoom
	
	:: Run some examples
	rem !PYTHON_EXECUTABLE! examples\python\basic.py
	rem !PYTHON_EXECUTABLE! examples\python\buffers.py
	rem !PYTHON_EXECUTABLE! examples\python\audio_buffer.py
	
	:: Run tests
	!PYTHON_EXECUTABLE! -m pytest tests\test_enums.py
	!PYTHON_EXECUTABLE! -m pytest tests\test_get_state.py
	!PYTHON_EXECUTABLE! -m pytest tests\test_make_action.py
	!PYTHON_EXECUTABLE! -m pytest tests\test_gym_wrapper.py
	
	!PYTHON_EXECUTABLE! -m pip uninstall -y vizdoom
)
