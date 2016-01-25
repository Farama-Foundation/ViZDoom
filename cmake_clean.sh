#!/bin/bash
for NAME in Makefile CMakeCache.txt CMakeFiles cmake_install.cmake
do
	find . -name "$NAME" -exec rm -rf {} 2>/dev/null \;
done
