#!/bin/bash
files="*.cpp"
for f in $files
do
  echo $f
  # sed -r "s/.*log/\/\//g" $f
done
