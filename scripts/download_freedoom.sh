#!/bin/sh

# Checks if bin/freedoom2.wad is in place if not, the zip is downloaded(if not yet present) and freedoom2.wad is extracted to bin directory.

FREEDOOM_LINK="https://github.com/freedoom/freedoom/releases/download/v0.10.1/freedoom-0.10.1.zip"

FREEDOOM_DOWNLOAD_PATH="./bin"
FREEDOOM_DESTINATION_PATH="."

if [ ! -e  "${FREEDOOM_DESTINATION_PATH}/freedoom2.wad" ]
then 
	if [ ! -e "${FREEDOOM_DOWNLOAD_PATH}/freedoom-0.10.1.zip" ]
	then
		wget --no-check-certificate ${FREEDOOM_LINK} -P ${FREEDOOM_DOWNLOAD_PATH}
	fi
	unzip -j -d ${FREEDOOM_DESTINATION_PATH} ${FREEDOOM_DOWNLOAD_PATH}/freedoom-0.10.1.zip freedoom-0.10.1/freedoom2.wad
fi

