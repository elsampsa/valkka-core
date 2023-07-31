/*
 * live_av_shmem_test.cpp : Test shmem and swscaling
 * 
 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>
 *
 */

/** 
 *  @file    live_av_shmem_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.5.1 
 *  
 *  @brief   Test shmem and swscaling
 *
 */

#include "framefifo.h"
#include "framefilter.h"
#include "logging.h"
#include "avdep.h"
#include "avthread.h"
#include "openglthread.h"
#include "livethread.h"
#include "sharedmem.h"
#include "test_import.h"
#include <sys/select.h>

using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char *stream_1 = std::getenv("VALKKA_TEST_RTSP_1");
const char *stream_2 = std::getenv("VALKKA_TEST_RTSP_2");
const char *stream_sdp = std::getenv("VALKKA_TEST_SDP");

void test_1()
{
    // (LiveThread:livethread) --> {FrameFilter:info} --> {FifoFrameFilter:in_filter} -->> (AVThread:avthread) --> {SwScaleFrameFilter:sw_scale} --> {InfoFrameFilter:scaled}
    InfoFrameFilter decoded_info("scaled");
    SwScaleFrameFilter sw_scale("sws_scale", 100, 100, &decoded_info);
    // InfoFrameFilter     decoded_info("decoded",&sws_scale);
    TimeIntervalFrameFilter
        interval("interval", 1000, &sw_scale); // pass a frame each 1000 ms
    AVThread avthread("avthread", interval);
    FifoFrameFilter &in_filter = avthread.getFrameFilter(); // request framefilter from AVThread
    InfoFrameFilter out_filter("encoded", &in_filter);
    LiveThread livethread("live");

    // return;
    const char *name = "@TEST: live_av_shmem_test: test 1: ";
    std::cout << name << "** @@YUV->RGB interpolation on the CPU **" << std::endl;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    std::cout << name << "starting threads" << std::endl;
    livethread.startCall();
    avthread.startCall();

    avthread.decodingOnCall();

    //sleep_for(2s);

    std::cout << name << "registering stream" << std::endl;
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx);

    // sleep_for(1s);

    std::cout << name << "playing stream !" << std::endl;
    livethread.playStreamCall(ctx);

    sleep_for(5s);
    // sleep_for(604800s); //one week

    std::cout << name << "stopping threads" << std::endl;
    livethread.stopCall();
    // avthread.  stopCall();
    // AVThread destructor => Thread destructor => stopCall (Thread::stopCall or AVThread::stopCall ..?) .. in destructor, virtual methods are not called
}

void test_2()
{
    // (LiveThread:livethread) --> {FrameFilter:info} --> {FifoFrameFilter:in_filter} -->> (AVThread:avthread) --> {SwScaleFrameFilter:sw_scale} --> {InfoFrameFilter:scaled} --> {RGBShmemFrameFilter:shmem}
    RGBShmemFrameFilter shmem("rgb_shmem", 10, 100, 100, 1000);
    InfoFrameFilter decoded_info("scaled", &shmem);
    SwScaleFrameFilter sw_scale("sws_scale", 100, 100, &decoded_info);
    // InfoFrameFilter     decoded_info("decoded",&sws_scale);
    TimeIntervalFrameFilter
        interval("interval", 1000, &sw_scale); // pass a frame each 1000 ms
    AVThread avthread("avthread", interval);
    FifoFrameFilter &in_filter = avthread.getFrameFilter(); // request framefilter from AVThread
    InfoFrameFilter out_filter("encoded", &in_filter);
    LiveThread livethread("live");

    const char *name = "@TEST: live_av_shmem_test: test 2: SERVER SIDE ";
    std::cout << name << "** @@YUV->RGB interpolation on the CPU, pass RGB frames to shared memory **" << std::endl;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    std::cout << name << "starting threads" << std::endl;
    livethread.startCall();
    avthread.startCall();

    avthread.decodingOnCall();

    // sleep_for(2s);

    std::cout << name << "registering stream" << std::endl;
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx);

    // sleep_for(1s);

    std::cout << name << "playing stream !" << std::endl;
    livethread.playStreamCall(ctx);

    sleep_for(10s);
    // sleep_for(5s);
    // sleep_for(604800s); //one week

    std::cout << name << "stopping threads" << std::endl;
    livethread.stopCall();
}

void test_3()
{
    const char *name = "@TEST: live_av_shmem_test: test 3: CLIENT SIDE ";
    std::cout << name << "** @@Read RGB frame data from the shared memory.  Start this after starting test 2.  Must be started separataly while the server-side is running**" << std::endl;

    SharedMemRingBufferRGB rb("rgb_shmem", 10, 100, 100, 1000, false);
    int index, n, ii;
    bool ok;

    while (true)
    {
        ok = rb.clientPull(index, n);
        if (ok)
        {
            std::cout << "cell index " << index << " has " << n << " bytes" << std::endl;
            for (ii = 0; ii < std::min(n, 10); ii++)
            {
                std::cout << int(rb.shmems[index]->payload[ii]) << std::endl;
            }
            std::cout << std::endl;
        }
        else
        {
            std::cout << "semaphore timed out!" << std::endl;
            std::cout << std::endl;
        }
    }
}

void test_4()
{
    // (LiveThread:livethread) --> {FrameFilter:info} --> {FifoFrameFilter:in_filter} -->> (AVThread:avthread) --> {SwScaleFrameFilter:sw_scale} --> {InfoFrameFilter:scaled} --> {RGBShmemFrameFilter:shmem}
    RGBShmemFrameFilter shmem("rgb_shmem", 10, 100, 100, 1000);
    InfoFrameFilter decoded_info("scaled", &shmem);
    SwScaleFrameFilter sw_scale("sws_scale", 300, 300, &decoded_info); // test sending oversized image
    // InfoFrameFilter     decoded_info("decoded",&sws_scale);
    TimeIntervalFrameFilter
        interval("interval", 1000, &sw_scale); // pass a frame each 1000 ms
    AVThread avthread("avthread", interval);
    FifoFrameFilter &in_filter = avthread.getFrameFilter(); // request framefilter from AVThread
    InfoFrameFilter out_filter("encoded", &in_filter);
    LiveThread livethread("live");

    const char *name = "@TEST: live_av_shmem_test: test 2: ";
    std::cout << name << "** @@YUV->RGB interpolation on the CPU, pass RGB frames to shared memory.  Use oversized images. **" << std::endl;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    std::cout << name << "starting threads" << std::endl;
    livethread.startCall();
    avthread.startCall();

    avthread.decodingOnCall();

    // sleep_for(2s);

    std::cout << name << "registering stream" << std::endl;
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx);

    // sleep_for(1s);

    std::cout << name << "playing stream !" << std::endl;
    livethread.playStreamCall(ctx);

    // sleep_for(10s);
    sleep_for(5s);
    // sleep_for(604800s); //one week

    std::cout << name << "stopping threads" << std::endl;
    livethread.stopCall();
}

void test_5()
{
    const char *name = "@TEST: live_av_shmem_test: test 5: Full server + client multiprocess with file descriptor & RGB shmem";
    std::cout << name << "** @@YUV->RGB interpolation on the CPU, pass RGB frames to shared memory ** " << std::endl;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    EventFd efd = EventFd(); // create the event file descriptor before fork

    int ni = 400;
    int nj = 500;
    int nmax = 1000;
    int nqueue = 10;

    int pid = fork();

    if (pid > 0) // in parent
    {
        // (LiveThread:livethread) --> {InfoFrameFilter:out_filter} --> {FifoFrameFilter:in_filter} -->>
        // (AVThread:avthread) --> {TimeIntervalFrameFilter:interval} --> {SwScaleFrameFilter:sw_scale} -->
        // {InfoFrameFilter:scaled} --> {RGBShmemFrameFilter:shmem}
        //
        RGBShmemFrameFilter shmem("rgb_shmem", nqueue, ni, nj, nmax);
        InfoFrameFilter decoded_info("scaled", &shmem);
        SwScaleFrameFilter sw_scale("sws_scale", ni, nj, &decoded_info);
        // InfoFrameFilter     decoded_info("decoded",&sws_scale);
        TimeIntervalFrameFilter
            interval("interval", 10, &sw_scale); // pass a frame each N ms
        AVThread avthread("avthread", interval);
        FifoFrameFilter &in_filter = avthread.getFrameFilter(); // request framefilter from AVThread
        InfoFrameFilter out_filter("encoded", &in_filter);
        LiveThread livethread("live");

        shmem.useFd(efd);

        std::cout << name << "starting threads" << std::endl;
        livethread.startCall();
        avthread.startCall();

        avthread.decodingOnCall();

        // sleep_for(2s);

        std::cout << name << "registering stream" << std::endl;
        LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info
        livethread.registerStreamCall(ctx);

        // sleep_for(1s);

        std::cout << name << "playing stream !" << std::endl;
        livethread.playStreamCall(ctx);

        sleep_for(10s);
        // sleep_for(5s);
        // sleep_for(3600s);
        // sleep_for(604800s); //one week

        std::cout << name << "stopping threads" << std::endl;
        livethread.stopCall();
    }
    else // in child
    {
        sleep_for(2s); // must use this in order to work with valgrind use 2+ secs
        // .. otherwise valgrind does some strange shit!  It doesn't reserve memory for the client:
        // SharedMemRingBufferRGB rb("rgb_shmem", nqueue, ni-10, nj-100, nmax, false);
        SharedMemRingBufferRGB rb("rgb_shmem", nqueue, ni, nj, nmax, false);
        RGB24Meta rgb_meta;
        rb.clientUseFd(efd);
        int index, n, ii;
        bool ok;
        int deadcount = 0;
        int retval;
        fd_set rfds;

        while (true)
        {
            // before pulling from client, do poll / select
            FD_ZERO(&rfds);
            FD_SET(efd.getFd(), &rfds);
            std::cout << "SELECT" << std::endl;
            retval = select(efd.getFd() + 1, &rfds, NULL, NULL, NULL);
            // ok = rb.clientPull(index, n);
            ok = rb.clientPullFrame(index, rgb_meta);
            if (ok)
            {
                std::cout << "CLIENT: size, w, h, slot, mstimestamp "
                          << rgb_meta.size << " "
                          << rgb_meta.width << " "
                          << rgb_meta.height << " "
                          << rgb_meta.slot << " "
                          << rgb_meta.mstimestamp << " "
                          << std::endl;
                for (ii = 0; ii < std::min(n, 10); ii++)
                {
                    std::cout << int(rb.shmems[index]->payload[ii]) << std::endl;
                }
                std::cout << std::endl;
            }
            else
            {
                std::cout << "CLIENT: semaphore timed out!" << std::endl;
                std::cout << std::endl;
                deadcount++;
            }
            if (deadcount >= 10)
            {
                break;
            }
        }
        std::cout << "CLIENT: exit" << std::endl;
    }
}

void test_6()
{
    const char *name = "@TEST: live_av_shmem_test: test 6: Full server + client multiprocess with file descriptor & vanilla shmem ";
    std::cout << name << "** @@YUV->RGB interpolation on the CPU, pass frames to shared memory **" << std::endl;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    // EventFd efd = EventFd(); // create the event file descriptor before fork

    int pid;

    pid = fork();
    // pid = 1;

    if (pid > 0) // in parent
    {
        // (LiveThread:livethread) --> {FrameFilter:info} --> {FifoFrameFilter:in_filter} -->> (AVThread:avthread) --> {SwScaleFrameFilter:sw_scale} --> {InfoFrameFilter:scaled} --> {RGBShmemFrameFilter:shmem}
        ShmemFrameFilter shmem("shmem", 10, 100, 1000); // 100 bytes: image overflows, but this is on purpose
        InfoFrameFilter info("info", &shmem);
        TimeIntervalFrameFilter
            interval("interval", 1, &info); // pass a frame each n ms
        InfoFrameFilter out_filter("encoded", &interval);
        LiveThread livethread("live");

        // shmem.useFd(efd);

        std::cout << name << "starting threads" << std::endl;
        livethread.startCall();
        /*
        avthread.  startCall();
        avthread.decodingOnCall();
        */

        // sleep_for(2s);

        std::cout << name << "registering stream" << std::endl;
        LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info
        livethread.registerStreamCall(ctx);

        // sleep_for(1s);

        std::cout << name << "playing stream !" << std::endl;
        livethread.playStreamCall(ctx);

        // sleep_for(10s);
        sleep_for(5s);
        // sleep_for(604800s); //one week

        std::cout << name << "stopping threads" << std::endl;
        livethread.stopCall();
    }
    else // in child
    {
        sleep_for(2s); // must use this in order to work with valgrind use 2+ secs
        // .. otherwise valgrind does some strange shit!  It doesn't reserve memory for the client:
        SharedMemRingBuffer rb("shmem", 10, 100, 1000, false);
        // rb.clientUseFd(efd);
        int index, n, ii;
        bool ok;
        int deadcount = 0;
        int retval;
        fd_set rfds;

        while (true)
        {
            // before pulling from client, do poll / select
            /*
            FD_ZERO(&rfds);
            FD_SET(efd.getFd(), &rfds);
            std::cout << "SELECT" << std::endl;
            retval = select(efd.getFd()+1, &rfds, NULL, NULL, NULL);
            */
            ok = rb.clientPull(index, n);
            if (ok)
            {
                std::cout << "CLIENT: cell index " << index << " has " << n << " bytes" << std::endl;
                for (ii = 0; ii < std::min(n, 10); ii++)
                {
                    std::cout << int(rb.shmems[index]->payload[ii]) << std::endl;
                }
                std::cout << std::endl;
            }
            else
            {
                std::cout << "CLIENT: semaphore timed out!" << std::endl;
                std::cout << std::endl;
                deadcount++;
            }
            if (deadcount >= 10)
            {
                break;
            }
        }
        std::cout << "CLIENT: exit" << std::endl;
    }
}

void test_7()
{
    const char *name = "@TEST: live_av_shmem_test: test 7: Full server + client multiprocess with RGB shmem";
    std::cout << name << "** @@YUV->RGB interpolation on the CPU, pass RGB frames to shared memory ** " << std::endl;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    // EventFd efd = EventFd(); // create the event file descriptor before fork

    int ni = 400;
    int nj = 500;
    int nmax = 1000;
    int nqueue = 10;

    int pid = fork();

    if (pid > 0) // in parent
    {
        // (LiveThread:livethread) --> {InfoFrameFilter:out_filter} --> {FifoFrameFilter:in_filter} -->>
        // (AVThread:avthread) --> {TimeIntervalFrameFilter:interval} --> {SwScaleFrameFilter:sw_scale} -->
        // {InfoFrameFilter:scaled} --> {RGBShmemFrameFilter:shmem}
        //
        RGBShmemFrameFilter shmem("rgb_shmem", nqueue, ni, nj, nmax);
        InfoFrameFilter decoded_info("scaled", &shmem);
        SwScaleFrameFilter sw_scale("sws_scale", ni, nj, &decoded_info);
        // InfoFrameFilter     decoded_info("decoded",&sws_scale);
        TimeIntervalFrameFilter
            interval("interval", 10, &sw_scale); // pass a frame each N ms
        AVThread avthread("avthread", interval);
        FifoFrameFilter &in_filter = avthread.getFrameFilter(); // request framefilter from AVThread
        InfoFrameFilter out_filter("encoded", &in_filter);
        LiveThread livethread("live");

        // shmem.useFd(efd);

        std::cout << name << "starting threads" << std::endl;
        livethread.startCall();
        avthread.startCall();

        avthread.decodingOnCall();

        // sleep_for(2s);

        std::cout << name << "registering stream" << std::endl;
        LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info
        livethread.registerStreamCall(ctx);

        // sleep_for(1s);

        std::cout << name << "playing stream !" << std::endl;
        livethread.playStreamCall(ctx);

        sleep_for(10s);
        // sleep_for(5s);
        // sleep_for(3600s);
        // sleep_for(604800s); //one week

        std::cout << name << "stopping threads" << std::endl;
        livethread.stopCall();
    }
    else // in child
    {
        sleep_for(2s); // must use this in order to work with valgrind use 2+ secs
        // .. otherwise valgrind does some strange shit!  It doesn't reserve memory for the client:
        // SharedMemRingBufferRGB rb("rgb_shmem", nqueue, ni-10, nj-100, nmax, false);
        SharedMemRingBufferRGB rb("rgb_shmem", nqueue, ni, nj, nmax, false);
        RGB24Meta rgb_meta;
        // rb.clientUseFd(efd);
        int index, n, ii;
        bool ok;
        int deadcount = 0;
        while (true)
        {
            ok = rb.clientPullFrame(index, rgb_meta);
            if (ok)
            {
                std::cout << "CLIENT: size, w, h, slot, mstimestamp "
                          << rgb_meta.size << " "
                          << rgb_meta.width << " "
                          << rgb_meta.height << " "
                          << rgb_meta.slot << " "
                          << rgb_meta.mstimestamp << " "
                          << std::endl;
                for (ii = 0; ii < rgb_meta.size; ii++)
                {
                    // std::cout << int(rb.shmems[index]->payload[ii]) << std::endl;
                }
                std::cout << std::endl;
            }
            else
            {
                std::cout << "CLIENT: semaphore timed out!" << std::endl;
                std::cout << std::endl;
                deadcount++;
            }
            if (deadcount >= 10)
            {
                break;
            }
        }
        std::cout << "CLIENT: exit" << std::endl;
    }
}

int main(int argc, char **argcv)
{
    if (argc < 2)
    {
        std::cout << argcv[0] << " needs an integer argument.  Second interger argument (optional) is verbosity" << std::endl;
    }
    else
    {

        if (argc > 2)
        { // choose verbosity
            switch (atoi(argcv[2]))
            {
            case (0): // shut up
                ffmpeg_av_log_set_level(0);
                fatal_log_all();
                break;
            case (1): // normal
                break;
            case (2): // more verbose
                ffmpeg_av_log_set_level(100);
                debug_log_all();
                break;
            case (3): // extremely verbose
                ffmpeg_av_log_set_level(100);
                crazy_log_all();
                break;
            default:
                std::cout << "Unknown verbosity level " << atoi(argcv[2]) << std::endl;
                exit(1);
                break;
            }
        }

        switch (atoi(argcv[1]))
        { // choose test
        case (1):
            test_1();
            break;
        case (2):
            test_2();
            break;
        case (3):
            test_3();
            break;
        case (4):
            test_4();
            break;
        case (5):
            test_5();
            break;
        case (6):
            test_6();
            break;
        case (7):
            test_7();
            break;
        default:
            std::cout << "No such test " << argcv[1] << " for " << argcv[0] << std::endl;
        }
    }
}

/*  Some useful code:


if (!stream_1) {
  std::cout << name <<"ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1"<< std::endl;
  exit(2);
}
std::cout << name <<"** test rtsp stream 1: "<< stream_1 << std::endl;
  
if (!stream_2) {
  std::cout << name <<"ERROR: missing test stream 2: set environment variable VALKKA_TEST_RTSP_2"<< std::endl;
  exit(2);
}
std::cout << name <<"** test rtsp stream 2: "<< stream_2 << std::endl;
    
if (!stream_sdp) {
  std::cout << name <<"ERROR: missing test sdp file: set environment variable VALKKA_TEST_SDP"<< std::endl;
  exit(2);
}
std::cout << name <<"** test sdp file: "<< stream_sdp << std::endl;

  
*/
