#!/bin/bash
#
# Includes both libValkka and python bindings in the python "source" package
#
# keep a softlink valkka/libValkka.so.* => distribution version of the library
#
# mod setup.py (see comments in that file and in README.md) and MANIFEST.in
sed -i -r "s/# ext_modules\=\[\] # SWITCH/ext_modules\=\[\] # SWITCH/g" setup.py
sed -i -r "s/# include valkka\/libValkka\.so\.\*/include valkka\/libValkka\.so\.\*/g" MANIFEST.in
python3 setup.py sdist
echo
echo Behold, your python package file:
echo
ls -l dist/*.tar.gz
echo
# recover setup.py and MANIFEST.in
sed -i -r "s/ext_modules\=\[\] # SWITCH/# ext_modules\=\[\] # SWITCH/g" setup.py
sed -i -r "s/include valkka\/libValkka\.so\.\*/# include valkka\/libValkka\.so\.\*/g" MANIFEST.in
