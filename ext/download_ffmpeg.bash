#!/bin/bash
release="3.4" # check that this is consistent with "configure_ffmpeg.bash"
rm -r -f ffmpeg
rm -f ffmpeg-$release.tar.gz
wget http://ffmpeg.org/releases/ffmpeg-$release.tar.gz
tar xvf ffmpeg-$release.tar.gz
mv ffmpeg-$release ffmpeg
