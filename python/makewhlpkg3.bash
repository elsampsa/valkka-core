#!/bin/bash
#
# I keep a softlink "valkka/libValkka.so.0 => distribution version of the library"
#
python3 setup.py bdist_wheel -p manylinux1_x86-64
echo
echo Behold, your python package file:
echo
ls -l dist/*.whl
echo
