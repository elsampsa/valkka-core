#!/bin/bash
python3 setup.py sdist
echo
echo Behold, your python package file:
echo
ls -l dist/*.tar.gz
echo
