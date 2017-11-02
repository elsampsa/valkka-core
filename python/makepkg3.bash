#!/bin/bash
python3 setup.py bdist_wheel
echo
echo Behold, your python package file:
echo
ls -l dist/*.whl
echo
