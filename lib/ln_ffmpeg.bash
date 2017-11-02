#!/bin/bash
if [ $# -ne 1 ]; then
  echo "Give ffmpeg root dir (root dir = directory with ffmpeg RELEASE and COPYING.* files)"
  exit
fi
basedir=$1
# # basedir=$HOME/ffmpeg/ffmpeg_git_lgpl # edit: the ffmpeg main directory, i.e. the directory where you have the "RELEASE" and "COPYING.*" files
# edit: for dynamic libraries, check your library versions!
# *** Dynamic libraries ***
ln -s $basedir/libavcodec/libavcodec.so.57
ln -s $basedir/libavformat/libavformat.so.57
ln -s $basedir/libavutil/libavutil.so.55
ln -s $basedir/libswscale/libswscale.so.4
# *** Static libraries ***
ln -s $basedir/libavcodec/libavcodec.a
ln -s $basedir/libavformat/libavformat.a
ln -s $basedir/libavutil/libavutil.a
ln -s $basedir/libswscale/libswscale.a
ln -s $basedir/libavfilter/libavfilter.a
ln -s $basedir/libavdevice/libavdevice.a
ln -s $basedir/libswresample/libswresample.a

# lib/libavcodec.a lib/libavformat.a lib/libavutil.a lib/libswscale.a lib/libavfilter.a lib/libavdevice.a lib/libswresample.a