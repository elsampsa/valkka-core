#!/bin/bash
let c=1
for f in *_test.cpp
do
  b="$(grep -E "brief.*" $f -o)"
  echo "if true; then"
  echo "# if false; then"
  echo "valgrind=\"valgrind\" # enable valgrind"
  echo "# valgrind=\"\" # disable valgrind"
  echo "echo $f"
  echo "# "$f" : "$b
  echo
  echo fi
  echo
done
