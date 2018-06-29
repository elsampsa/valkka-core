#!/bin/bash
echo
echo "USING PYTHON HEADERS FROM:"
python3 -c "from distutils import sysconfig; print(sysconfig.get_python_lib())"
echo
echo "LOADING NUMPY FROM:"
python3 -c "import numpy; print(numpy.get_include())"
echo
echo "NUMPY VERSION:"
python3 -c "import numpy; print(numpy.__version__)"
echo
