#!/bin/bash
files="libavformat.a libavutil.a libswscale.a libavfilter.a libavdevice.a libswresample.a"
for file in $files
do
  echo $file
  nm $file  | grep $1
done
