#!/bin/bash
if [ $# -lt 2 ]; then
  echo "1: Give version, i.e: 1.2.3"
  echo "2: comment"
  exit
fi
cp changelog .changelog
ver=$1
shift
# exit
# distros="bionic focal jammy"
distros="focal jammy"
for distro in $distros
do
echo
echo $distro $ver $@
echo
echo "PREPENDING CHANGELOG & CREATING SCRIPT FOR "$distro
echo
read -r
./addlog.py $ver $distro $@ | cat - changelog > changelog_tmp
mv changelog_tmp changelog
echo
echo "RUNNING SCRIPT FOR "$distro
echo
read -r
./rundeb.bash
done
echo "REMOVING CHANGELOG SAVE .changelog"
read -r
rm .changelog
