#!/bin/bash
python3 setup.py bdist_wheel -p manylinux1_x86-64
echo
echo Behold, your python package file:
echo
ls -l dist/*.whl
echo
