#!/bin/bash
#
# Downloads exteral libraries (live555 and fffmpeg) and prepares them
#
cd ext
./download_live.bash
./download_ffmpeg.bash
cd ..
make -f debian/rules clean
