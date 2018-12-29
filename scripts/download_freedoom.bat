@echo off
setlocal enabledelayedexpansion

:: Checks if bin/freedoom2.wad is in place if not, the zip is downloaded (if not yet present) and freedoom2.wad is extracted to bin directory.

:: Older version of freedoom
::set FREEDOOM_LINK="https://github.com/freedoom/freedoom/releases/download/v0.10.1/freedoom-0.10.1.zip"
set FREEDOOM_LINK="https://github.com/freedoom/freedoom/releases/download/v0.11.3/freedoom-0.11.3.zip"
set FREEDOOM_ARCHIVE=%FREEDOOM_LINK:~0,-1%
for %%F in (%FREEDOOM_LINK%) do set FREEDOOM_ARCHIVE=%%~nxF

set FREEDOOM_OUTFILE=".\%FREEDOOM_ARCHIVE%"
set FREEDOOM_DESTINATION_PATH=".\bin"
set FREEDOOM_DESTINATION_FILE="%FREEDOOM_DESTINATION_PATH%\freedoom2.wad"

if not exist "%FREEDOOM_DESTINATION_FILE%" (
	if not exist "%FREEDOOM_DESTINATION_PATH%" md "%FREEDOOM_DESTINATION_PATH%"
    if not exist "%FREEDOOM_OUTFILE%" (
        powershell -command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest '%FREEDOOM_LINK%' -OutFile '%FREEDOOM_OUTFILE%'"
	)
    powershell -command "Expand-Archive '%FREEDOOM_OUTFILE%' -DestinationPath '%FREEDOOM_DESTINATION_PATH%'"
    copy "%FREEDOOM_DESTINATION_PATH%\%FREEDOOM_ARCHIVE:~0,-4%\freedoom2.wad" "%FREEDOOM_DESTINATION_FILE%"
    rd "%FREEDOOM_DESTINATION_PATH%\%FREEDOOM_ARCHIVE:~0,-4%" /S /Q
)
