/*
 * av_live_thread_test.cpp : Test producer (live thread) consumer (av thread)
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
 *  @file    av_live_thread_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.3.0 
 *  
 *  @brief Test producer (live thread) consumer (av thread)
 *
 */

#include "framefifo.h"
#include "livethread.h"
#include "avthread.h"
#include "framefilter.h"
#include "framefilter2.h"
#include "logging.h"
#include "test_import.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for; // http://en.cppreference.com/w/cpp/thread/sleep_for

const char *stream_1 = std::getenv("VALKKA_TEST_RTSP_1");
const char *stream_2 = std::getenv("VALKKA_TEST_RTSP_2");
const char *stream_sdp = std::getenv("VALKKA_TEST_SDP");

// (LiveThread:livethread) --> {FrameFilter:info} --> {FifoFrameFilter:in_filter} -->> (AVThread:avthread) --> {InfoFrameFilter:decoded_info}
InfoFrameFilter decoded_info("decoded");
AVThread avthread("avthread", decoded_info);
FifoFrameFilter &in_filter = avthread.getFrameFilter(); // request framefilter from AVThread
InfoFrameFilter out_filter("encoded", &in_filter);
LiveThread livethread("live");

void test_1()
{
    const char *name = "@TEST: av_live_thread_test: test 1: ";
    std::cout << name << "** @@Send frames from live to av thread **" << std::endl;

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

    sleep_for(2s);

    std::cout << name << "registering stream" << std::endl;
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx);

    sleep_for(1s);

    std::cout << name << "playing stream !" << std::endl;
    livethread.playStreamCall(ctx);

    sleep_for(10s);
    // sleep_for(604800s); //one week

    std::cout << name << "stopping threads" << std::endl;
    livethread.stopCall();
    // avthread.  stopCall();
    // AVThread destructor => Thread destructor => stopCall (Thread::stopCall or AVThread::stopCall ..?) .. in destructor, virtual methods are not called
    //
}

void test_2()
{
    const char *name = "@TEST: av_live_thread_test: test 2: ";
    std::cout << name << "** @@Send frames from live to av thread.  Short **" << std::endl;

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

    std::cout << name << "registering stream" << std::endl;
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx);

    std::cout << name << "playing stream !" << std::endl;
    livethread.playStreamCall(ctx);

    sleep_for(3s);
    // sleep_for(604800s); //one week

    std::cout << name << "stopping threads" << std::endl;
    // livethread.stopCall(); // TODO: when not calling this .. valgrind says that LiveConnectionContext.msreconnect not initialized
}

void test_3()
{
    const char *name = "@TEST: av_live_thread_test: test 3: ";
    std::cout << name << "** @@Send frames from live to av thread.  Test effect of frames running out. TODO **" << std::endl;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    // TODO
}

void test_4()
{
    const char *name = "@TEST: av_live_thread_test: test 4: ";
    std::cout << name << "** @@Send frames from live to av thread.  Long run. **" << std::endl;

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

    std::cout << name << "registering stream" << std::endl;
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx);

    std::cout << name << "playing stream !" << std::endl;
    livethread.playStreamCall(ctx);

    // sleep_for(5s);
    sleep_for(604800s); //one week

    std::cout << name << "stopping threads" << std::endl;
}

void test_5()
{
    const char *name = "@TEST: av_live_thread_test: test 5: ";
    std::cout << name << "** @@Send frames from live to av thread.  Test multithreading decoding. **" << std::endl;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    avthread.setNumberOfThreads(4);

    std::cout << name << "starting threads" << std::endl;
    livethread.startCall();
    avthread.startCall();

    avthread.decodingOnCall();

    std::cout << name << "registering stream" << std::endl;
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx);

    std::cout << name << "playing stream !" << std::endl;
    livethread.playStreamCall(ctx);

    sleep_for(5s);
    // sleep_for(60s);
    // sleep_for(604800s); //one week

    std::cout << name << "stopping threads" << std::endl;
}

void test_6()
{
    const char *name = "@TEST: av_live_thread_test: test 6: ";
    std::cout << name << "** @@Send frames from live to av thread.  Test sdp source & exotic bitmaps **" << std::endl;

    if (!stream_sdp)
    {
        std::cout << name << "ERROR: missing test sdp stream: set environment variable VALKKA_TEST_SDP" << std::endl;
        exit(2);
    }
    std::cout << name << "** test sdp stream: " << stream_sdp << std::endl;

    avthread.setNumberOfThreads(1);

    std::cout << name << "starting threads" << std::endl;
    livethread.startCall();
    avthread.startCall();

    avthread.decodingOnCall();

    std::cout << name << "registering stream" << std::endl;
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::sdp, std::string(stream_sdp), 2, &out_filter); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx);

    std::cout << name << "playing stream !" << std::endl;
    livethread.playStreamCall(ctx);

    sleep_for(5s);
    // sleep_for(60s);
    // sleep_for(604800s); //one week

    std::cout << name << "stopping threads" << std::endl;
}


void test_7()
{
    DumpAVBitmapFrameFilter dumpbm("dumpbm");
    AVThread avthread("avthread", dumpbm);
    FifoFrameFilter &in_filter = avthread.getFrameFilter(); // request framefilter from AVThread
    InfoFrameFilter out_filter("encoded", &in_filter);
    LiveThread livethread("live");

    const char *name = "@TEST: av_live_thread_test: test 7: ";
    std::cout << name << "** @@Send frames from live to av thread to a framefilter that dumps YUV frames onto disk**" << std::endl;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    avthread.setNumberOfThreads(4);

    std::cout << name << "starting threads" << std::endl;
    livethread.startCall();
    avthread.startCall();

    avthread.decodingOnCall();

    std::cout << name << "registering stream" << std::endl;
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx);

    std::cout << name << "playing stream !" << std::endl;
    livethread.playStreamCall(ctx);

    sleep_for(2s);
    // sleep_for(60s);
    // sleep_for(604800s); //one week

    std::cout << name << "stopping threads" << std::endl;
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
