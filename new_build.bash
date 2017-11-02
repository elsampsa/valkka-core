#!/bin/bash
if [ $# -ne 1 ]; then
  echo "Give a build name!"
  exit
fi
dirname="build_"$1
echo "Your build is in "$dirname
mkdir -p $dirname
cp tools/build/checkdep.py $dirname
cp tools/build/check_pkgs.bash $dirname
cp tools/build/clean*.bash $dirname
cp tools/build/run_cmake.bash $dirname
cp tools/build/test_env.bash $dirname
cp tools/build/README_* $dirname
cp tools/build/set_test_streams.bash $dirname
