#!/usr/bin/env bash

# Checks if bin/freedoom2.wad is in place if not, the zip is downloaded (if not yet present) and freedoom2.wad is extracted to bin directory.

# Older version of freedoom
#FREEDOOM_LINK="https://github.com/freedoom/freedoom/releases/download/v0.10.1/freedoom-0.10.1.zip"
FREEDOOM_LINK="https://github.com/freedoom/freedoom/releases/download/v0.11.3/freedoom-0.11.3.zip"
FREEDOOM_ARCHIVE=$(echo ${FREEDOOM_LINK} | cut -d '/' -f9)
FREEDOOM_ARCHIVE_BASENAME=$(basename ${FREEDOOM_ARCHIVE} .zip)

FREEDOOM_DOWNLOAD_PATH="."
FREEDOOM_DESTINATION_PATH="./bin"

if [ ! -e  "${FREEDOOM_DESTINATION_PATH}/freedoom2.wad" ]; then
	if [ ! -e "${FREEDOOM_DOWNLOAD_PATH}/${FREEDOOM_ARCHIVE}" ]; then
		wget --no-check-certificate ${FREEDOOM_LINK} -P ${FREEDOOM_DOWNLOAD_PATH} -O $FREEDOOM_ARCHIVE
	fi
	unzip -j -d ${FREEDOOM_DESTINATION_PATH} ${FREEDOOM_DOWNLOAD_PATH}/${FREEDOOM_ARCHIVE} ${FREEDOOM_ARCHIVE_BASENAME}/freedoom2.wad
fi
