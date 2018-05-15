#!/bin/bash
#
# I keep a softlink "valkka/libValkka.so.0 => distribution version of the library"
#
python3 setup.py sdist
echo
echo Behold, your python package file:
echo
ls -l dist/*.tar.gz
echo
