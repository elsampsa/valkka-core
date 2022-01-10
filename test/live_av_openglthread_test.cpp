/*
 * live_av_openglthread_test.cpp : Test the full pipeline: LiveThread => AVThread => OpenGLThread
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
 *  @file    live_av_openglthread_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.3.0 
 *  
 *  @brief   Test the full pipeline: LiveThread => AVThread => OpenGLThread
 *
 */

#include "framefifo.h"
#include "framefilter.h"
#include "logging.h"
#include "avdep.h"
#include "avthread.h"
#include "openglthread.h"
#include "livethread.h"
#include "test_import.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char *stream_1 = std::getenv("VALKKA_TEST_RTSP_1");
const char *stream_2 = std::getenv("VALKKA_TEST_RTSP_2");
const char *stream_sdp = std::getenv("VALKKA_TEST_SDP");

// (LiveThread:livethread) --> {FrameFilter:info} --> {FifoFrameFilter:in_filter} -->> (AVThread:avthread) --> {InfoFrameFilter:decoded_info} -->> (OpenGLThread:glthread)

// OpenGLThread    glthread("gl_thread");
OpenGLThread glthread("gl_thread", OpenGLFrameFifoContext(), DEFAULT_OPENGLTHREAD_BUFFERING_TIME, ":0.0");
// OpenGLThread    glthread("gl_thread", OpenGLFrameFifoContext(), DEFAULT_OPENGLTHREAD_BUFFERING_TIME, ":0.1"); // test seconds x-screen .. this actually works! :)
// OpenGLThread    glthread("gl_thread", OpenGLFrameFifoContext(), DEFAULT_OPENGLTHREAD_BUFFERING_TIME, ":1.0"); // nopes .. only one connection allowed?

FifoFrameFilter &gl_in_filter = glthread.getFrameFilter();
// InfoFrameFilter decoded_info("decoded",&gl_in_filter);
DummyFrameFilter decoded_info("decoded", false, &gl_in_filter); // non-verbose
AVThread avthread("avthread", decoded_info);
FifoFrameFilter &in_filter = avthread.getFrameFilter(); // request framefilter from AVThread
// InfoFrameFilter out_filter("encoded",&in_filter);
DummyFrameFilter out_filter("encoded", false, &in_filter); // non-verbose
LiveThread livethread("live");

void test_1()
{
    const char *name = "@TEST: live_av_openglthread_test: test 1: feed frames with no render context ";
    std::cout << name << "** @@Feeding frames with no render context => frames are just being queued. **" << std::endl;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    std::cout << name << "starting threads" << std::endl;
    glthread.startCall();
    avthread.startCall();
    livethread.startCall();

    avthread.decodingOnCall();

    sleep_for(2s);

    std::cout << name << "registering stream" << std::endl;
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx);

    sleep_for(1s);

    std::cout << name << "playing stream !" << std::endl;
    livethread.playStreamCall(ctx);

    sleep_for(5s);
    // sleep_for(604800s); //one week

    std::cout << name << "stopping threads" << std::endl;
    livethread.stopCall();
    avthread.stopCall();
    glthread.stopCall();
    // AVThread destructor => Thread destructor => stopCall (Thread::stopCall or AVThread::stopCall ..?) .. in destructor, virtual methods are not called
}

void test_2()
{
    const char *name = "@TEST: live_av_openglthread_test: test 2: ";
    std::cout << name << "** @@Test window & context creation, etc. **" << std::endl;
    int i;

    // start glthread and create a window
    glthread.startCall();
    Window window_id = glthread.createWindow();
    glthread.makeCurrent(window_id);
    std::cout << "new x window " << window_id << std::endl;

    // create render group & context
    glthread.newRenderGroupCall(window_id);
    i = glthread.newRenderContextCall(2, window_id, 0);
    std::cout << "got render context id " << i << std::endl;

    // del render group & context
    glthread.delRenderContextCall(i);
    glthread.delRenderGroupCall(window_id);

    std::cout << name << "stopping threads" << std::endl;
    glthread.stopCall();
}

void test_3()
{
    const char *name = "@TEST: live_av_openglthread_test: test 3: ";
    std::cout << name << "** @@Feed one rtsp stream to a single window **" << std::endl;
    int i;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    std::cout << name << "starting threads" << std::endl;

    // avthread.  setTimeTolerance(10); // 10 milliseconds // just testing ..

    avthread.setNumberOfThreads(1);

    glthread.setStaticTexFile("1.yuv");

    // start glthread and create a window
    glthread.startCall();
    Window window_id = glthread.createWindow();
    glthread.makeCurrent(window_id);
    std::cout << "new x window " << window_id << std::endl;

    // start av and live threads
    avthread.startCall();
    livethread.startCall();
    avthread.decodingOnCall(); // don't forget this ..
    sleep_for(2s);

    std::cout << name << "registering stream" << std::endl;
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx);
    std::cout << name << "playing stream !" << std::endl;
    livethread.playStreamCall(ctx);

    // create render group & context
    glthread.newRenderGroupCall(window_id);
    i = glthread.newRenderContextCall(2, window_id, 0);
    std::cout << "got render context id " << i << std::endl;

    sleep_for(10s);
    // sleep_for(30s);
    // sleep_for(120s);
    // sleep_for(604800s); //one week

    // del render group & context
    glthread.delRenderContextCall(i);
    glthread.delRenderGroupCall(window_id);

    std::cout << name << "stopping threads" << std::endl;
    livethread.stopCall();
    avthread.stopCall();
    glthread.stopCall();
}

void test_4()
{
    const char *name = "@TEST: live_av_openglthread_test: test 4: ";
    std::cout << name << "** @@Feed one rtsp stream to two x windows **" << std::endl;
    int i1, i2;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    std::cout << name << "starting threads" << std::endl;

    // start glthread and create two windows
    glthread.startCall();

    Window window_id1 = glthread.createWindow();
    glthread.makeCurrent(window_id1);
    std::cout << "new x window 1" << window_id1 << std::endl;

    Window window_id2 = glthread.createWindow();
    glthread.makeCurrent(window_id2);
    std::cout << "new x window 2" << window_id2 << std::endl;

    // start av and live threads
    avthread.startCall();
    livethread.startCall();
    avthread.decodingOnCall(); // don't forget this ..
    sleep_for(2s);

    std::cout << name << "registering stream" << std::endl;
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info

    // ctx.time_correction = TimeCorrectionType::none;

    livethread.registerStreamCall(ctx);
    std::cout << name << "playing stream !" << std::endl;
    livethread.playStreamCall(ctx);

    // create render group & context
    glthread.newRenderGroupCall(window_id1);
    i1 = glthread.newRenderContextCall(2, window_id1, 0);
    std::cout << "got render context id 1 " << i1 << std::endl;

    glthread.newRenderGroupCall(window_id2);
    i2 = glthread.newRenderContextCall(2, window_id2, 0);
    std::cout << "got render context id 2 " << i2 << std::endl;

    sleep_for(5s);
    // sleep_for(30s);
    // sleep_for(604800s); //one week

    // del render group & context
    glthread.delRenderContextCall(i1);
    glthread.delRenderContextCall(i2);
    glthread.delRenderGroupCall(window_id1);
    glthread.delRenderGroupCall(window_id2);

    std::cout << name << "stopping threads" << std::endl;
    livethread.stopCall();
    avthread.stopCall();
    glthread.stopCall();
}

void test_5()
{
    const char *name = "@TEST: live_av_openglthread_test: test 5: ";
    std::cout << name << "** @@Feed two rtsp streams to x windows **" << std::endl;
    int i1, i2;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    if (!stream_2)
    {
        std::cout << name << "ERROR: missing test stream 2: set environment variable VALKKA_TEST_RTSP_2" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 2: " << stream_2 << std::endl;

    AVThread avthread1("avthread1", gl_in_filter);
    FifoFrameFilter &in_filter_1 = avthread1.getFrameFilter(); // request framefilter from AVThread
    AVThread avthread2("avthread2", gl_in_filter);
    FifoFrameFilter &in_filter_2 = avthread2.getFrameFilter(); // request framefilter from AVThread

    std::cout << name << "starting threads" << std::endl;
    // start glthread and create two windows
    glthread.startCall();

    Window window_id1 = glthread.createWindow();
    glthread.makeCurrent(window_id1);
    std::cout << "new x window 1" << window_id1 << std::endl;

    Window window_id2 = glthread.createWindow();
    glthread.makeCurrent(window_id2);
    std::cout << "new x window 2" << window_id2 << std::endl;

    // start av and live threads
    avthread1.startCall();
    avthread2.startCall();
    livethread.startCall();
    avthread1.decodingOnCall(); // don't forget this ..
    avthread2.decodingOnCall(); // don't forget this ..
    sleep_for(2s);

    std::cout << name << "registering streams" << std::endl;
    LiveConnectionContext ctx1 = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &in_filter_1); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx1);
    LiveConnectionContext ctx2 = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_2), 3, &in_filter_2); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx2);

    std::cout << name << "playing streams" << std::endl;
    livethread.playStreamCall(ctx1);
    livethread.playStreamCall(ctx2);

    // create render group & context
    glthread.newRenderGroupCall(window_id1);
    i1 = glthread.newRenderContextCall(2, window_id1, 0);
    std::cout << "got render context id 1 " << i1 << std::endl;

    glthread.newRenderGroupCall(window_id2);
    i2 = glthread.newRenderContextCall(3, window_id2, 0);
    std::cout << "got render context id 2 " << i2 << std::endl;

    sleep_for(5s);
    // sleep_for(10s);
    // sleep_for(604800s); //one week

    // del render group & context
    glthread.delRenderContextCall(i1);
    glthread.delRenderContextCall(i2);
    glthread.delRenderGroupCall(window_id1);
    glthread.delRenderGroupCall(window_id2);

    std::cout << name << "stopping threads" << std::endl;
    livethread.stopCall();
    avthread1.stopCall();
    avthread2.stopCall();
    glthread.stopCall();
}

void test_6()
{

    const char *name = "@TEST: live_av_openglthread_test: test 6: ";
    std::cout << name << "** @@Send one stream to several windows **" << std::endl;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    std::cout << name << "starting threads" << std::endl;

    // avthread.  setTimeTolerance(10); // 10 milliseconds // just testing ..

    // glthread.setStaticTexFile("1.yuv");

    // start glthread and create a window
    glthread.startCall();

    // start av and live threads
    avthread.startCall();
    livethread.startCall();
    avthread.decodingOnCall(); // don't forget this ..
    sleep_for(2s);

    std::cout << name << "registering stream" << std::endl;
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx);
    std::cout << name << "playing stream !" << std::endl;

    std::vector<Window> windows;
    int n = 5;
    // int n=1;
    int i;

    // /* // context per window
    for (i = 0; i <= n; i++)
    {
        std::cout << i << std::endl;
        Window window_id = glthread.createWindow();
        glthread.makeCurrent(window_id);
        std::cout << "new x window " << window_id << std::endl;
        windows.push_back(window_id);
    }

    sleep_for(1s);

    for (auto it = windows.begin(); it != windows.end(); it++)
    {
        // create render group & context
        glthread.newRenderGroupCall(*it);
        int ii = glthread.newRenderContextCall(2, *it, 0);
        std::cout << "got render context id " << ii << std::endl;
    }
    // */

    /* // all contexes to same window
  Window window_id=glthread.createWindow();
  glthread.newRenderGroupCall(window_id);
  
  for(i=0; i<=n; i++) {
    int ii=glthread.newRenderContextCall(2, window_id, 0);
    std::cout << "got render context id "<<ii<<std::endl;
  }
  */

    /* // various tests
  for(i=0;i<=n;i++) {
    std::cout << i << std::endl;
    Window window_id=glthread.createWindow();
    glthread.makeCurrent(window_id);
    std::cout << "new x window "<<window_id<<std::endl;
    windows.push_back(window_id);
  }
  
  sleep_for(1s);
  
  for(auto it=windows.begin(); it!=windows.end(); it++) {
    // create render group
    glthread.newRenderGroupCall(*it);
    sleep_for(1s);
  }
  
  for(auto it=windows.begin(); it!=windows.end(); it++) {
    // create render context
    // int ii=glthread.newRenderContextCall(2, *it, 0); // same slot // crasssh
    int ii=glthread.newRenderContextCall(3, *it, 0); // dead slot // crasssh
    // int ii=glthread.newRenderContextCall(3, 9999, 0); // fake render groups (windows) => render context discarded
    std::cout << "got render context id "<<ii<<std::endl;
    sleep_for(1s);
  }
  */

    livethread.playStreamCall(ctx);

    sleep_for(5s);
    // sleep_for(10s);
    // sleep_for(120s);
    // sleep_for(604800s); //one week

    std::cout << name << "stopping threads" << std::endl;
    livethread.stopCall();
    avthread.stopCall();
    glthread.stopCall();
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
