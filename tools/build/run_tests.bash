#!/bin/bash
#
# This file was created with the aid of "gen_sections.bash" and "gen_scripts" in the "tests/" directory
#
# Enable/disable test sections by un/commenting the if true/if false lines
# Some of the sections need special, manual intervention (see the "NOTICE" texts)
#
# Do the following on the output file:
#
# grep "ERROR SUMMARY" bin/test.out
# grep -i "invalid" bin/test.out | grep "=="
# grep "Syscall" bin/test.out | grep "=="

# grep -i "warning" bin/test.out
# grep "definitely lost" bin/test.out
# grep "indirectly lost" bin/test.out
#
# The segfaults (if any) will appear on the screen as you run the tests.  Check also manually bin/test.out
#
LOGLEVEL=1
echo
echo Performing tests with the following cameras:
echo
echo "VALKKA_TEST_RTSP_1 = "$VALKKA_TEST_RTSP_1
echo "VALKKA_TEST_RTSP_2 = "$VALKKA_TEST_RTSP_2
echo "VALKKA_TEST_SDP    = "$VALKKA_TEST_SDP
echo
echo "loglevel           = "$LOGLEVEL
echo
read -p "Press any key to continue... " -n1 -s
echo
echo "Running"
echo
cd bin
rm -f test.out


# if false; then
if true; then
echo "frame_test"
# frame_test.cpp : brief Testing Frame classes. Compile with "make tests" and run with valgrind
# @@Create frame **
echo 1
valgrind ./frame_test 1 $LOGLEVEL &>> test.out
printf "END: frame_test 1\n\n" &>> test.out

# @@Frame filters **
echo 2
valgrind ./frame_test 2 $LOGLEVEL &>> test.out
printf "END: frame_test 2\n\n" &>> test.out

# @@Frame filters **
echo 4
valgrind ./frame_test 4 $LOGLEVEL &>> test.out
printf "END: frame_test 4\n\n" &>> test.out

fi

# if false; then
if true; then
echo "thread_test"
# thread_test.cpp : brief Testing the Thread class using Test*Thread classes. Compile with "make tests" and run with valgrind
# @@Consumer and producer **
echo 1
valgrind ./thread_test 1 $LOGLEVEL &>> test.out
printf "END: thread_test 1\n\n" &>> test.out

# @@Consumer and producer 2 **
echo 2
valgrind ./thread_test 2 $LOGLEVEL &>> test.out
printf "END: thread_test 2\n\n" &>> test.out

fi


# if false; then
if true; then
echo "avthread_test"
# avthread_test.cpp : brief Test class AVThread
# @@Send a setup frame to AVThread **
echo 1
valgrind ./avthread_test 1 $LOGLEVEL &>> test.out
printf "END: avthread_test 1\n\n" &>> test.out

# @@Send a setup frame and two void video frames to AVThread **
echo 2
valgrind ./avthread_test 2 $LOGLEVEL &>> test.out
printf "END: avthread_test 2\n\n" &>> test.out

# @@Send a void video frame to AVThread (no setup frame) **
echo 3
valgrind ./avthread_test 3 $LOGLEVEL &>> test.out
printf "END: avthread_test 3\n\n" &>> test.out

# @@Send two consecutive setup frames to AVThread (i.e. reinit) **
echo 4
valgrind ./avthread_test 4 $LOGLEVEL &>> test.out
printf "END: avthread_test 4\n\n" &>> test.out

# @@Start two AVThreads **
echo 5
valgrind ./avthread_test 5 $LOGLEVEL &>> test.out
printf "END: avthread_test 5\n\n" &>> test.out

# @@Send setup, video and audio frames to AVThread **
echo 6
valgrind ./avthread_test 6 $LOGLEVEL &>> test.out
printf "END: avthread_test 6\n\n" &>> test.out

fi


# if false; then
if true; then
echo "framefifo_test"
# framefifo_test.cpp : brief Testing fifo classes. Compile with "make tests" and run with valgrind
# @@Basic fifo tests **
echo 1
valgrind ./framefifo_test 1 $LOGLEVEL &>> test.out
printf "END: framefifo_test 1\n\n" &>> test.out

# @@Fifo overflow (no flush)**
echo 2
valgrind ./framefifo_test 2 $LOGLEVEL &>> test.out
printf "END: framefifo_test 2\n\n" &>> test.out

# @@Fifo overflow (with flush)**
echo 3
valgrind ./framefifo_test 3 $LOGLEVEL &>> test.out
printf "END: framefifo_test 3\n\n" &>> test.out

# @@Fifo overflow (with flush) : diagnosis output **
echo 4
valgrind ./framefifo_test 4 $LOGLEVEL &>> test.out
printf "END: framefifo_test 4\n\n" &>> test.out

# @@Basic fifo tests: with different frame classes **
echo 5
valgrind ./framefifo_test 5 $LOGLEVEL &>> test.out
printf "END: framefifo_test 5\n\n" &>> test.out

fi

# if false; then
if true; then
echo "framefilter_test"
# framefilter_test.cpp : brief Testing some (more complex) FrameFilters
# @@Test ForkFrameFilterN **
echo 1
valgrind ./framefilter_test 1 $LOGLEVEL &>> test.out
printf "END: framefilter_test 1\n\n" &>> test.out

# @@DESCRIPTION **
echo 2
valgrind ./framefilter_test 2 $LOGLEVEL &>> test.out
printf "END: framefilter_test 2\n\n" &>> test.out

# @@DESCRIPTION **
echo 3
valgrind ./framefilter_test 3 $LOGLEVEL &>> test.out
printf "END: framefilter_test 3\n\n" &>> test.out

# @@DESCRIPTION **
echo 4
valgrind ./framefilter_test 4 $LOGLEVEL &>> test.out
printf "END: framefilter_test 4\n\n" &>> test.out

echo 5
# @@DESCRIPTION **
valgrind ./framefilter_test 5 $LOGLEVEL &>> test.out
printf "END: framefilter_test 5\n\n" &>> test.out

fi


# if false; then
if true; then
echo "livethread_test"
echo "NOTICE: for some subtests, you need to run aux/unicast_to_multicast.bash script before initiating this test"
# livethread_test.cpp : brief Testing the LiveThread class
echo 1
valgrind ./livethread_test 1 $LOGLEVEL &>> test.out
printf "END: livethread_test 1\n\n" &>> test.out

# @@Print payload, one rtsp connection **
echo 2
valgrind ./livethread_test 2 $LOGLEVEL &>> test.out
printf "END: livethread_test 2\n\n" &>> test.out

# @@Inspect stream from a rtsp connection for 10 secs **
echo 3
valgrind ./livethread_test 3 $LOGLEVEL &>> test.out
printf "END: livethread_test 3\n\n" &>> test.out

# @@Starting and stopping, single sdp connection **
echo 4
valgrind ./livethread_test 4 $LOGLEVEL &>> test.out
printf "END: livethread_test 4\n\n" &>> test.out

# @@Two connections and filters, starting and stopping, two rtsp connections **
echo 5
valgrind ./livethread_test 5 $LOGLEVEL &>> test.out
printf "END: livethread_test 5\n\n" &>> test.out

# @@Starting and stopping, various sdp connections **
echo 6
valgrind ./livethread_test 6 $LOGLEVEL &>> test.out
printf "END: livethread_test 6\n\n" &>> test.out

# @@Starting and stopping, single rtsp connection.  Testing triggerEvent. **
echo 7
valgrind ./livethread_test 7 $LOGLEVEL &>> test.out
printf "END: livethread_test 7\n\n" &>> test.out

# @@Sending frames between LiveThreads**
echo 8
valgrind ./livethread_test 8 $LOGLEVEL &>> test.out
printf "END: livethread_test 8\n\n" &>> test.out

# @@Feeding frames back to LiveThread**
echo 9
valgrind ./livethread_test 9 $LOGLEVEL &>> test.out
printf "END: livethread_test 9\n\n" &>> test.out

# @@Sending frames between LiveThreads: short time test **
echo 10
valgrind ./livethread_test 10 $LOGLEVEL &>> test.out
printf "END: livethread_test 10\n\n" &>> test.out

# @@Print payload, one rtsp connection.  Test auto reconnection. **
echo 11
valgrind ./livethread_test 11 $LOGLEVEL &>> test.out
printf "END: livethread_test 11\n\n" &>> test.out
#
# NOTICE: let's not do this..
# # @@Sending frames between LiveThreads: long time test **
# echo 12
# valgrind ./livethread_test 12 $LOGLEVEL &>> test.out
fi


# if false; then
if true; then
echo "livethread_rtsp_test"
# livethread_rtsp_test.cpp : brief
# @@Test rtsp server**
echo 1
valgrind ./livethread_rtsp_test 1 $LOGLEVEL &>> test.out
printf "END: livethread_rtsp_test 1\n\n" &>> test.out

# @@Sending frames from client to rtsp server**
echo 2
valgrind ./livethread_rtsp_test 2 $LOGLEVEL &>> test.out
printf "END: livethread_rtsp_test 2\n\n" &>> test.out

# @@Sending frames from client to rtsp server**
echo 3
valgrind ./livethread_rtsp_test 3 $LOGLEVEL &>> test.out
printf "END: livethread_rtsp_test 3\n\n" &>> test.out

# @@DESCRIPTION **
echo 4
valgrind ./livethread_rtsp_test 4 $LOGLEVEL &>> test.out
printf "END: livethread_rtsp_test 4\n\n" &>> test.out

# @@DESCRIPTION **
echo 5
valgrind ./livethread_rtsp_test 5 $LOGLEVEL &>> test.out
printf "END: livethread_rtsp_test 5\n\n" &>> test.out
fi


# if false; then
if true; then
echo "switch_test"
# switch_test.cpp : brief Test the Switch and DoubleGate classes
# @@Stream from two cameras, switch between the two **
echo 1
valgrind ./switch_test 1 $LOGLEVEL &>> test.out
printf "END: switch_test 1\n\n" &>> test.out

# @@Stream and decode from two cameras, switch between the two **
echo 2
valgrind ./switch_test 2 $LOGLEVEL &>> test.out
printf "END: switch_test 2\n\n" &>> test.out

# NOTE: let's not do this
echo "NOTICE: to run test 3 with valgrind, build your code with VALGRIND_DEBUG enabled.  Run this also manually (without valgrind) and see if you get video"
echo "bin/switch_test 3 0"
# # @@Stream, decode and present from two cameras, switch between the two **
# echo 3
# valgrind ./switch_test 3 $LOGLEVEL &>> test.out
# printf "END: switch_test 3\n\n" &>> test.out

# @@DESCRIPTION **
echo 4
valgrind ./switch_test 4 $LOGLEVEL &>> test.out
printf "END: switch_test 4\n\n" &>> test.out

# @@DESCRIPTION **
echo 5
valgrind ./switch_test 5 $LOGLEVEL &>> test.out
printf "END: switch_test 5\n\n" &>> test.out
fi


# if false; then
if true; then
echo "file_test"
echo "NOTICE: produces file kokkelis.mkv"
# file_test.cpp : brief Test file input
# @@Test FileFrameFilter : produces kokkelis.mkv **
echo 1
valgrind ./file_test 1 $LOGLEVEL &>> test.out
printf "END: file_test 1\n\n" &>> test.out

# @@Test FileFrameFilter: wrong file **
echo 2
valgrind ./file_test 2 $LOGLEVEL &>> test.out
printf "END: file_test 2\n\n" &>> test.out

# @@Test FileFrameFilter: activate, deActivate : produces kokkelis.mkv **
echo 3
valgrind ./file_test 3 $LOGLEVEL &>> test.out
printf "END: file_test 3\n\n" &>> test.out

# @@Test FileFrameFilter: dirty close : produces kokkelis.mkv  **
echo 4
valgrind ./file_test 4 $LOGLEVEL &>> test.out
printf "END: file_test 4\n\n" &>> test.out

# @@DESCRIPTION **
echo 5
valgrind ./file_test 5 $LOGLEVEL &>> test.out
printf "END: file_test 5\n\n" &>> test.out
fi


# if false; then
if true; then
echo "filethread_test"
echo "NOTICE: these tests must be run either (i) with valgrind and VALGRIND_DEBUG enabled or (ii) without valgrind and with visual inspection"
echo "It's better to run these tests each one individually from command line (they seek, pause and play) and see if the video works"
# valgrind="valgrind" # enable valgrind
valgrind="" # disable valgrind
# filethread_test.cpp : brief Test file input
# @@Test FileThread **
echo 1
$valgrind ./filethread_test 1 $LOGLEVEL &>> test.out
printf "END: filethread_test 1\n\n" &>> test.out

# @@Stream from FileThread to AVThread and OpenGLThread. Seek. **
echo 2
$valgrind ./filethread_test 2 $LOGLEVEL &>> test.out
printf "END: filethread_test 2\n\n" &>> test.out

# @@Stream from FileThread to AVThread and OpenGLThread. Seek, play, stop, etc.**
echo 3
$valgrind ./filethread_test 3 $LOGLEVEL &>> test.out
printf "END: filethread_test 3\n\n" &>> test.out

# @@Stream from FileThread to AVThread and OpenGLThread.  Two streams.  Seek, play, stop, etc.**
echo 4
$valgrind ./filethread_test 4 $LOGLEVEL &>> test.out
printf "END: filethread_test 4\n\n" &>> test.out

# @@Stream from FileThread to AVThread and OpenGLThread. Seek and play. **
echo 5
$valgrind ./filethread_test 5 $LOGLEVEL &>> test.out
printf "END: filethread_test 5\n\n" &>> test.out

# @@Stream from FileThread to AVThread. Seek, play, stop, etc.**
echo 6
$valgrind ./filethread_test 6 $LOGLEVEL &>> test.out
printf "END: filethread_test 6\n\n" &>> test.out

fi


# if false; then
if true; then
echo "av_live_thread_test"

# @@Send frames from live to av thread **
echo 1
valgrind ./av_live_thread_test 1 $LOGLEVEL &>> test.out
printf "END: av_live_thread_test 1\n\n" &>> test.out

# @@Send frames from live to av thread.  Short **
echo 2
valgrind ./av_live_thread_test 2 $LOGLEVEL &>> test.out
printf "END: av_live_thread_test 2\n\n" &>> test.out

# @@Send frames from live to av thread.  Test effect of frames running out. TODO **
echo 3
valgrind ./av_live_thread_test 3 $LOGLEVEL &>> test.out
printf "END: av_live_thread_test 3\n\n" &>> test.out

# NOTICE: let's not do this
# # @@Send frames from live to av thread.  Long run. **
# echo 4
# valgrind ./av_live_thread_test 4 $LOGLEVEL &>> test.out
# printf "END: av_live_thread_test 4\n\n" &>> test.out

# @@ Test reading from SPD source, using yuv422p bitmap
echo 6
echo WARNING: for this test to work, you must start 
echo the script unicast_to_multicast_usb.bash
valgrind ./av_live_thread_test 6 $LOGLEVEL &>> test.out
printf "END: av_live_thread_test 3\n\n" &>> test.out

fi


if false; then
# if true; then
echo "shmem_test"
echo "NOTICE: These are interactive test where you should start two terminals, one for the shmem server and other one for the client"
echo "Refer to the test source code how to use them"
echo
## @@Create shared memory on the server side : INTERACTIVE TEST **
#echo 1
#valgrind ./shmem_test 1 $LOGLEVEL &>> test.out
#printf "END: shmem_test 1\n\n" &>> test.out

## @@Create shared memory on the client side : INTERACTIVE TEST **
#echo 2
#valgrind ./shmem_test 2 $LOGLEVEL &>> test.out
#printf "END: shmem_test 2\n\n" &>> test.out

## @@Create shmem ring buffer on the server side : INTERACTIVE TEST **
#echo 3
#valgrind ./shmem_test 3 $LOGLEVEL &>> test.out
#printf "END: shmem_test 3\n\n" &>> test.out

## @@Create shmem ring buffer on the client side : INTERACTIVE TEST **
#echo 4
#valgrind ./shmem_test 4 $LOGLEVEL &>> test.out
#printf "END: shmem_test 4\n\n" &>> test.out

## @@DESCRIPTION : TODO **
#echo 5
#valgrind ./shmem_test 5 $LOGLEVEL &>> test.out
#printf "END: shmem_test 5\n\n" &>> test.out

## @@Test ShmemFrameFilter **
#echo 6
#valgrind ./shmem_test 6 $LOGLEVEL &>> test.out
#printf "END: shmem_test 6\n\n" &>> test.out

fi

if true; then
echo "shmem_test - part 2"
echo 8
valgrind ./shmem_test 8 $LOGLEVEL &>> test.out
printf "END: shmem_test 8\n\n" &>> test.out
fi

# if false; then
if true; then
echo "ringbuffer_test"
echo "non-interactive shmem ringbuffer test"
valgrind="valgrind" # enable valgrind
# valgrind="" # disable valgrind

echo 1
$valgrind ./ringbuffer_test 1 $LOGLEVEL &>> test.out
printf "END: ringbuffer_test 1\n\n" &>> test.out

echo 2
$valgrind ./ringbuffer_test 2 $LOGLEVEL &>> test.out
printf "END: ringbuffer_test 2\n\n" &>> test.out

fi

# if false; then
if true; then
echo "live_av_shmem_test"
echo "NOTICE: These are interactive test where you should start two terminals, one for the shmem server and other one for the client"
echo "Refer to the test source code how to use them"
echo "Here we just test the server side"
echo

# @@YUV->RGB interpolation on the CPU **
echo 1
valgrind ./live_av_shmem_test 1 $LOGLEVEL &>> test.out
printf "END: live_av_shmem_test 1\n\n" &>> test.out

# @@YUV->RGB interpolation on the CPU, pass RGB frames to shared memory **
echo 2
valgrind ./live_av_shmem_test 2 $LOGLEVEL &>> test.out
printf "END: live_av_shmem_test 2\n\n" &>> test.out

## @@Read RGB frame data from the shared memory.  Start this after starting test 2.  Must be started separataly while the server-side is running**
#echo 3
#valgrind ./live_av_shmem_test 3 $LOGLEVEL &>> test.out
# printf "END: live_av_shmem_test 3\n\n" &>> test.out

# @@DESCRIPTION **
echo 4
valgrind ./live_av_shmem_test 4 $LOGLEVEL &>> test.out
printf "END: live_av_shmem_test 4\n\n" &>> test.out

# @Full multiprocess test with server & client for rgb shmem frames **
echo 5
valgrind ./live_av_shmem_test 5 $LOGLEVEL &>> test.out
printf "END: live_av_shmem_test 5\n\n" &>> test.out

# @Full multiprocess test with server & client for vanilla shmem frames **
echo 6
valgrind ./live_av_shmem_test 6 $LOGLEVEL &>> test.out
printf "END: live_av_shmem_test 6\n\n" &>> test.out

echo 6
valgrind ./live_av_shmem_test 7 $LOGLEVEL &>> test.out
printf "END: live_av_shmem_test 7\n\n" &>> test.out

fi

if true; then
echo "live_muxshmem_test"
echo "testing frag-mp4 muxing and related shmem API in the same go"
echo

echo 2
valgrind ./live_muxshmem_test 2 $LOGLEVEL &>> test.out
printf "END: live_av_shmem_test 1\n\n" &>> test.out

fi


# if false; then
if true; then
echo "live_av_openglthread_test"
echo "NOTICE: these tests must be run either (i) with valgrind and VALGRIND_DEBUG enabled or (ii) without valgrind and with visual inspection"
# valgrind="valgrind" # enable valgrind
valgrind="" # disable valgrind

# live_av_openglthread_test.cpp : brief Test the full pipeline: LiveThread => AVThread => OpenGLThread
# @@Feeding frames with no render context => frames are just being queued. **
echo 1
$valgrind ./live_av_openglthread_test 1 $LOGLEVEL &>> test.out
printf "END: live_av_openglthread_test 1\n\n" &>> test.out

# @@Test window & context creation, etc. **
echo 2
$valgrind ./live_av_openglthread_test 2 $LOGLEVEL &>> test.out
printf "END: live_av_openglthread_test 2\n\n" &>> test.out

# @@Feed one rtsp stream to a single window **
echo 3
$valgrind ./live_av_openglthread_test 3 $LOGLEVEL &>> test.out
printf "END: live_av_openglthread_test 3\n\n" &>> test.out

# @@Feed one rtsp stream to two x windows **
echo 4
$valgrind ./live_av_openglthread_test 4 $LOGLEVEL &>> test.out
printf "END: live_av_openglthread_test 4\n\n" &>> test.out

# @@Feed two rtsp streams to x windows **
echo 5
$valgrind ./live_av_openglthread_test 5 $LOGLEVEL &>> test.out
printf "END: live_av_openglthread_test 5\n\n" &>> test.out

# @@Send one stream to several windows **
echo 6
$valgrind ./live_av_openglthread_test 6 $LOGLEVEL &>> test.out
printf "END: live_av_openglthread_test 6\n\n" &>> test.out
fi


if true; then
echo "live_av_openglthread_test2"
echo "NOTICE: these tests must be run either (i) with valgrind and VALGRIND_DEBUG enabled or (ii) without valgrind and with visual inspection"
# valgrind="valgrind" # enable valgrind
valgrind="" # disable valgrind

echo 1
$valgrind ./live_av_openglthread_test2 1 $LOGLEVEL &>> test.out
printf "END: live_av_openglthread_test2 1\n\n" &>> test.out

echo 2
$valgrind ./live_av_openglthread_test2 2 $LOGLEVEL &>> test.out
printf "END: live_av_openglthread_test2 2\n\n" &>> test.out

echo 3
$valgrind ./live_av_openglthread_test2 3 $LOGLEVEL &>> test.out
printf "END: live_av_openglthread_test2 3\n\n" &>> test.out
fi


# if false; then
if true; then
echo "openglframefifo_test"
echo "NOTICE: this needs VALGRIND_DEBUG enabled.  Otherwise run without valgrind"
# valgrind="valgrind" # enable valgrind
valgrind="" # disable valgrind

# openglframefifo_test.cpp : brief
# @@DESCRIPTION **
echo 1
$valgrind ./openglframefifo_test 1 $LOGLEVEL &>> test.out
printf "END: openglframefifo_test 1\n\n" &>> test.out

# @@DESCRIPTION **
echo 2
$valgrind ./openglframefifo_test 2 $LOGLEVEL &>> test.out
printf "END: openglframefifo_test 2\n\n" &>> test.out

# @@DESCRIPTION **
echo 3
$valgrind ./openglframefifo_test 3 $LOGLEVEL &>> test.out
printf "END: openglframefifo_test 3\n\n" &>> test.out

# @@DESCRIPTION **
echo 4
$valgrind ./openglframefifo_test 4 $LOGLEVEL &>> test.out
printf "END: openglframefifo_test 4\n\n" &>> test.out

# @@DESCRIPTION **
echo 5
$valgrind ./openglframefifo_test 5 $LOGLEVEL &>> test.out
printf "END: openglframefifo_test 5\n\n" &>> test.out

fi


# if false; then
if true; then
echo "openglthread_test"
echo "NOTICE: this needs VALGRIND_DEBUG enabled.  Otherwise run without valgrind"
# valgrind="valgrind" # enable valgrind
valgrind="" # disable valgrind

# openglthread_test.cpp : brief start/stop the OpenGLThread etc.
# @@Test preRun and postRun **
echo 1
$valgrind ./openglthread_test 1 $LOGLEVEL &>> test.out
printf "END: openglthread_test 1\n\n" &>> test.out

# @@Test starting and stopping the thread **
echo 2
$valgrind ./openglthread_test 2 $LOGLEVEL &>> test.out
printf "END: openglthread_test 2\n\n" &>> test.out

# @@Start OpenGLThread for two separate X-screens/GPUs **
echo 3
$valgrind ./openglthread_test 3 $LOGLEVEL &>> test.out
printf "END: openglthread_test 3\n\n" &>> test.out

# @@DESCRIPTION **
echo 4
$valgrind ./openglthread_test 4 $LOGLEVEL &>> test.out
printf "END: openglthread_test 4\n\n" &>> test.out

# @@DESCRIPTION **
echo 5
$valgrind ./openglthread_test 5 $LOGLEVEL &>> test.out
printf "END: openglthread_test 5\n\n" &>> test.out

fi


# if false; then
if true; then
echo "usbthread_test"
valgrind="valgrind" # enable valgrind
# valgrind="" # disable valgrind

echo 3
$valgrind ./usbthread_test 3 $LOGLEVEL &>> test.out
printf "END: usbthread_test 3\n\n" &>> test.out

fi


# *** ValkkaFS tests ***

# if false; then
if true; then
echo "valkkafswriter_test"
valgrind="valgrind" # enable valgrind
# valgrind="" # disable valgrind

echo 3
$valgrind ./valkkafswriter_test 3 $LOGLEVEL &>> test.out
printf "END: valkkafswriter_test 3\n\n" &>> test.out

echo 4
$valgrind ./valkkafswriter_test 4 $LOGLEVEL &>> test.out
printf "END: valkkafswriter_test 4\n\n" &>> test.out
fi


if true; then
echo "cache_test"
valgrind="valgrind" # enable valgrind
# valgrind="" # disable valgrind

echo 1
$valgrind ./cache_test 1 $LOGLEVEL &>> test.out
printf "END: cache_test 1\n\n" &>> test.out

echo 2
$valgrind ./cache_test 2 $LOGLEVEL &>> test.out
printf "END: cache_test 2\n\n" &>> test.out
fi


if true; then
echo "cachestream_test"
valgrind="valgrind" # enable valgrind
# valgrind="" # disable valgrind

echo 1
$valgrind ./cachestream_test 1 $LOGLEVEL &>> test.out
printf "END: cachestream_test 1\n\n" &>> test.out

echo 4
$valgrind ./cachestream_test 4 $LOGLEVEL &>> test.out
printf "END: cachestream_test 4\n\n" &>> test.out
fi

grep "ERROR SUMMARY" test.out
grep "seg" test.out

