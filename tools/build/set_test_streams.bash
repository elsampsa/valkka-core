#!/bin/bash
# # Make a copy of this script and edit for your particular case
# # Run with source

# # RTPS 1
# export VALKKA_TEST_RTSP_1=rtsp://admin:12345@192.168.0.157 # works
export VALKKA_TEST_RTSP_1=rtsp://admin:123456@192.168.0.134 # works
# export VALKKA_TEST_RTSP_1=rtsp://admin:nordic12345@192.168.0.24 # works

# # RTPS 2
# export VALKKA_TEST_RTSP_2=rtsp://admin:12345@192.168.0.157
export VALKKA_TEST_RTSP_2=rtsp://admin:nordic12345@192.168.0.24 # works

# # SDP
export VALKKA_TEST_SDP=$PWD/aux/multicast.sdp
