#!/bin/bash
sed -r "s/FDWrite/$1/g" write_thread.h > tmp
sed -r "s/fd\_write/$2/g" tmp > $1_thread.h

sed -r "s/FDWrite/$1/g" write_thread.cpp > tmp
sed -r "s/fd\_write/$2/g" tmp > $1_thread.cpp

rm tmp
