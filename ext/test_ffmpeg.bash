#!/bin/bash
echo
echo setting LD_LIBRARY_PATH so that you can test the 
echo ffmpeg executable in directory ffmpeg/
echo remember to run this script with source!
echo
cd ffmpeg
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/libavcodec:$PWD/libavfilter:$PWD/libavutil:$PWD/libavformat:$PWD/libavdevice:$PWD/libswresample:$PWD/libswscale
echo
echo "now you can run ./ffmpeg"
echo "why not try this and test vaapi acceleration:"
echo "LIBVA_DRIVER_NAME=i965 ./ffmpeg -hwaccel vaapi -i rtsp://admin:123456@10.0.0.3 -c:v rawvideo -pix_fmt yuv420p out.yuv"
echo
