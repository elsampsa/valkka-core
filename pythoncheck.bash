#!/bin/bash
#echo
#echo "PYTHON3 PACKAGES IN:"
#python3 -c "from distutils import sysconfig; print(sysconfig.get_python_lib())"
echo
echo "USING NUMPY HEADERS FROM:"
python3 -c "import numpy; print(numpy.get_include())"
echo
echo "NUMPY VERSION:"
python3 -c "import numpy; print(numpy.__version__)"
echo
