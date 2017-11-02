#!/bin/bash
if [ $# -ne 2 ]; then
  echo "Give a string and its replacement"
  exit
fi

echo "Replacing "$1" with "$2
echo "Are you sure?"
read -n1 -r -p "Press q to quit, space to continue..." key

if [ "$key" = '' ]; then
  echo "running sed"
  find include/ -name "*.h" -exec sed -i -r "s/$1/$2/g" {} \;
  find src/ -name "*.cpp" -exec sed -i -r "s/$1/$2/g" {} \;
  find test/ -name "*.cpp" -exec sed -i -r "s/$1/$2/g" {} \;
else
  echo
  echo "cancelled"
fi
