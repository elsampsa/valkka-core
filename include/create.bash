#!/bin/bash
hg=$1"_HEADER_GUARD"
echo "#ifndef "$hg > $1.h
echo "#define "$hg >> $1.h
#define SIZES_HEADER_GUARD
st="s/NAME/"$1"/g"
cat template | sed $st >> $1.h
echo "#endif" >> $1.h

