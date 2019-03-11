#!/usr/bin/env bash

python3 setup.py sdist
NEWEST_PACKAGE=`ls dist/* | sort | tail -n 1`

twine upload ${NEWEST_PACKAGE}
