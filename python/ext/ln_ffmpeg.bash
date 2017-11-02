#!/bin/bash
if [ $# -ne 1 ]; then
  echo "Give ffmpeg root dir (root dir = directory with ffmpeg RELEASE and COPYING.* files)"
  exit
fi
basedir=$1
dirs="libavcodec libavformat libavutil libswscale libavfilter libavdevice libswresample"
for d in $dirs
do
  ln -s $basedir/$d .
done
