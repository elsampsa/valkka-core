#!/bin/bash
if [ $# -lt 2 ]; then
  echo "1: Give version, i.e: 1.2.3"
  echo "2: comment"
  exit
fi
# create file "control" dynamically, based on the host architecture
./control.sh
cp changelog .changelog
ver=$1
shift
# exit
# bionic: 18.04.6 LTS / focal: 20.04.6 LTS / jammy: 22.04.2 LTS
# distros="bionic focal jammy"
distros="focal jammy"
for distro in $distros
do
echo
echo $distro $ver $@
echo
echo "PREPENDING CHANGELOG & CREATING SCRIPT FOR "$distro
echo "PRESS ENTER"
echo
read -r
./addlog.py $ver $distro $@ | cat - changelog > changelog_tmp
mv changelog_tmp changelog
echo
echo "RUNNING SCRIPT FOR "$distro
echo "WARNING: WAIT UNTIL THE PASSWORD IS ASKED!"
echo "PRESS ENTER"
echo
read -r
./rundeb.bash
done
echo "REMOVING CHANGELOG SAVE .changelog"
echo "PRESS ENTER"
read -r
rm .changelog
rm control
