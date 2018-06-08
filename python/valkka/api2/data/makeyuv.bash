#!/bin/bash
# you need to install imagemagick to do this
# image resolution should be 1280x720
name=valkka_bw_logo
convert valkka_bw_logo.png valkka_bw_logo.yuv
identify valkka_bw_logo.png
# Now we now the dimensions and bit-depth.  Test:
identify -size 1280x720 -depth 8 valkka_bw_logo.yuv
