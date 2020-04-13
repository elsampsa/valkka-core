#!/bin/bash
# # Make a copy of this script and edit for your particular case

form="udp://"
target="224.1.168.91:50000" # multicast comes from here
src=$1

## h264 capable webcam:
# ffmpeg -y -c:v h264 -i /dev/video2 -c:v copy -an kokkelis.mkv
## bitmap one:
# ffmpeg -i /dev/video0 -c:v h264 -an kokkelis.mkv

# com="ffmpeg -i "$src" -r 10 -c:v h264 -preset veryfast -x264-params keyint=10:min-keyint=10 -an -map 0:0 -f rtp "$form""$target

## Baseline profile: Only P-frames, OK
# com="ffmpeg -i "$src" -r 20 -c:v h264 -pix_fmt yuv420p -profile:v baseline -preset veryfast -x264-params keyint=10:min-keyint=10 -an -map 0:0 -f rtp "$form""$target

## Main profile with B-frames: OK
# com="ffmpeg -i "$src" -r 10 -c:v h264 -pix_fmt yuv420p -profile:v main -preset veryfast -x264-params keyint=10:min-keyint=10 -an -map 0:0 -f rtp "$form""$target

## High profile: OK
# com="ffmpeg -i "$src" -r 10 -c:v h264 -pix_fmt yuv420p -profile:v high -preset veryfast -x264-params keyint=10:min-keyint=10 -an -map 0:0 -f rtp "$form""$target

## High profile with 422: OK
com="ffmpeg -i "$src" -r 10 -c:v h264 -pix_fmt yuv422p -profile:v high422 -preset veryfast -x264-params keyint=10:min-keyint=10 -an -map 0:0 -f rtp "$form""$target
# com="ffmpeg -i "$src" -r 10 -c:v h264 -preset veryfast -x264-params keyint=10:min-keyint=10 -an -map 0:0 -f rtp "$form""$target

## TODO: fix PTS timestamping for B-frame streams .. now just copying the PTS from the input in libValkka.

echo
echo $com
echo
$com

