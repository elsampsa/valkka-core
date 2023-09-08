/*
 * live_av_framefilter_test.cpp : Test some framefilters with the complete decoding pipeline
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
 *  @file    live_av_framefilter_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2023
 *  @version 1.5.4 
 *  
 *  @brief   Test some framefilters with the complete decoding pipeline
 *
 */

#include "framefifo.h"
#include "framefilter.h"
#include "framefilter2.h"
#include "logging.h"
#include "avdep.h"
#include "avthread.h"
#include "openglthread.h"
#include "livethread.h"
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
    // BriefInfoFrameFilter info("info");
    // FPSCountFrameFilter fps("cam 1", 1000, &info); // print fps every 1 sec
    FPSCountFrameFilter fps("cam 1", 1000); // print fps every 1 sec
    AVThread avthread("avthread", fps);
    FifoFrameFilter &av_filter = avthread.getFrameFilter(); // request framefilter from AVThread
    LiveThread livethread("live");

    // return;
    const char *name = "@TEST: live_av_framefilter_test: test 1: ";
    std::cout << name << "** YUV frames FPS test **" << std::endl;

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
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &av_filter); // Request livethread to write into filter info
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
}

void test_3()
{
}

void test_4()
{
}

void test_5()
{
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
        default:
            std::cout << "No such test " << argcv[1] << " for " << argcv[0] << std::endl;
        }
    }
}
