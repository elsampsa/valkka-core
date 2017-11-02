#!/bin/bash 
touch valkka_core.i
python3 setup.py build_ext
# copy python and .so file into the module directory
cp build/lib.linux-x86_64-3.5/_valkka_core.cpython-35m-x86_64-linux-gnu.so valkka/_valkka_core.so
cp valkka_core.py valkka/
