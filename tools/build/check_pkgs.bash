#!/bin/bash
for f in *.deb
do
  echo
  echo ------ $f ---------
  echo
  dpkg -c $f
  echo
  echo
  dpkg --info $f
  echo
  echo
  md5sum $f
  echo
done
