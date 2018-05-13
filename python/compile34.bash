#!/bin/bash 
./make_swig_file.bash
python3 setup.py build_ext
# copy python and .so file into the module directory
cp build/lib.linux-x86_64-3.4/_valkka_core.cpython-34m.so valkka/_valkka_core.so
cp valkka_core.py valkka/
