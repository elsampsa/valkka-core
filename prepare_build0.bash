#!/bin/bash
#
# Downloads exteral libraries - only ffmpeg
#
cd ext
# ./download_live.bash
./download_ffmpeg.bash 
cd ..
make -f debian/rules clean
