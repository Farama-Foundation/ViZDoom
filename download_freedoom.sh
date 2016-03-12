#!/bin/sh
# Checks if scenarios/freedoom2.wad is in place and downloads and extracts it if not.

FREEDOM_LINK="https://github.com/freedoom/freedoom/releases/download/v0.10.1/freedoom-0.10.1.zip"

if [ ! -e  "./scenarios/freedoom2.wad" ]
then 
	if [ ! -e "./bin/freedoom-0.10.1.zip" ]
	then
		wget $FREEDOM_LINK -P ./bin
	fi
	unzip -j -d ./scenarios ./bin/freedoom-0.10.1.zip freedoom-0.10.1/freedoom2.wad
fi

