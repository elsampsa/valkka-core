#!/bin/bash
if [ $# -ne 1 ]; then
  echo "Give live555 root dir (root dir = directory with ffmpeg RELEASE and COPYING files)"
  exit
fi
basedir=$1
ln -s $basedir/BasicUsageEnvironment/include ./BasicUsageEnvironment
ln -s $basedir/UsageEnvironment/include ./UsageEnvironment
ln -s $basedir/groupsock/include ./groupsock
ln -s $basedir/liveMedia/include ./liveMedia
