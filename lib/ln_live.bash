#!/bin/bash
if [ $# -ne 1 ]; then
  echo "Give live555 root dir (root dir = directory with ffmpeg RELEASE and COPYING files)"
  exit
fi
basedir=$1
# basedir=$HOME/live555/live/ # edit: the live555 main directory, i.e. the directory where you have the README and COPYING files
# edit: for dynamic libraries, check your library versions!
# *** Dynamic libraries ***
ln -s $basedir/BasicUsageEnvironment/libBasicUsageEnvironment.so.1.0.0 ./libBasicUsageEnvironment.so.1
ln -s $basedir/UsageEnvironment/libUsageEnvironment.so.3.1.0 ./libUsageEnvironment.so.3
ln -s $basedir/groupsock/libgroupsock.so.8.1.0 ./libgroupsock.so.8
ln -s $basedir/liveMedia/libliveMedia.so.52.3.3 ./libliveMedia.so.52
# *** Static libraries ***
# some notes here:
# At Live555, when creating the makefiles with "genMakefiles <architecture>", use "linux-64bit", because:
# config.linux-64bit : creates position independent code (PIC)
# config.linux       : no -fPIC here!
ln -s $basedir/BasicUsageEnvironment/libBasicUsageEnvironment.a
ln -s $basedir/UsageEnvironment/libUsageEnvironment.a
ln -s $basedir/groupsock/libgroupsock.a
ln -s $basedir/liveMedia/libliveMedia.a
