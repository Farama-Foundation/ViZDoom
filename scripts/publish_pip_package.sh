#!/usr/bin/env bash

python setup.py sdist
NEWEST_PACKAGE=`ls dist/* | sort | tail -n 1`
REPO="pypi"
twine register ${NEWEST_PACKAGE} -r ${REPO}
twine upload ${NEWEST_PACKAGE} -r ${REPO}

