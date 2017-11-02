#!/bin/bash
# this scripts creates a nice list of tests you can merge into CMakeLists.txt
ls -1 *.cpp | awk 'BEGIN{st=" \""} {split($0,a,"."); st=st a[1] "\" \""} END{print(st)}'
