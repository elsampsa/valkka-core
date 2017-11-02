#!/bin/bash
#
# list of all header files (from here, pick just some)
#
echo "%{ // this is prepended in the wapper-generated c(pp) file"
echo "#define SWIG_FILE_WITH_INIT"
find . -name "*.h" | sed -e "s/\.\//#include \"include\//" | awk '{print $0"\""}'
echo "%}"
echo
echo "// next, expose what is necessary"

