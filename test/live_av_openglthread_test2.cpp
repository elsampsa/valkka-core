/*
 * live_av_openglthread_test2.cpp : Test the full pipeline: LiveThread => AVThread => OpenGLThread .. and draw some boxes!
 * 
 * Copyright 2017, 2018 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>
 *
 */

/** 
 *  @file    live_av_openglthread_test2.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.17.0 
 *  
 *  @brief   Test the full pipeline: LiveThread => AVThread => OpenGLThread .. and draw some boxes!
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
DummyFrameFilter out_filter("decoded", false, &in_filter); // non-verbose
LiveThread livethread("live");

void test_1()
{
    const char *name = "@TEST: live_av_openglthread_test: test 1: ";
    std::cout << name << "** @@Feed one rtsp stream to a single window and draw overlaying boxes**" << std::endl;
    int i;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    std::cout << name << "starting threads" << std::endl;

    // avthread.  setTimeTolerance(10); // 10 milliseconds // just testing ..

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

    // glthread.addRectangleCall(i, -0.5, 0.5, 0.5, -0.5);
    // glthread.addRectangleCall(i, -0.1, 0.1, 0.25, -0.25);

    glthread.addRectangleCall(i, 0.25, 0.75, 0.75, 0.25); // left, right, top, bottom in 0..1 coordinates
    glthread.addRectangleCall(i, 0.1, 0.2, 0.75, 0.25);

    sleep_for(10s);

    glthread.clearObjectsCall(i);
    // glthread.addRectangleCall(i, -0.3, 0.3, 0.25, -0.25);
    glthread.addRectangleCall(i, 0.3, 0.9, 0.9, 0.1);

    sleep_for(5s);

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

void test_2()
{
    const char *name = "@TEST: live_av_openglthread_test2: test 2: ";
    std::cout << name << "** @@Create and delete several render contexes.  No video. **" << std::endl;

    // start glthread and create a window
    glthread.startCall();
    Window window_id = glthread.createWindow();
    glthread.makeCurrent(window_id);
    std::cout << "new x window " << window_id << std::endl;

    // create render group & context
    glthread.newRenderGroupCall(window_id);

    int n, i;
    for (n = 0; n <= 8; n++)
    {
        std::cout << "create " << n + 1 << std::endl;
        i = glthread.newRenderContextCall(2, window_id, 0);
        std::cout << "got render context id " << i << std::endl;
        // del render group & context
        sleep_for(2s);
        std::cout << "delete " << n + 1 << std::endl;
        glthread.delRenderContextCall(i);
    }

    glthread.delRenderGroupCall(window_id);

    std::cout << name << "stopping threads" << std::endl;
    glthread.stopCall();
}

void test_3()
{
    const char *name = "@TEST: live_av_openglthread_test2: test 3: ";
    std::cout << name << "** @@Create and delete several render contexes **" << std::endl;

    int n, i;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    std::cout << name << "starting threads" << std::endl;

    // avthread.  setTimeTolerance(10); // 10 milliseconds // just testing ..

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
    std::cout << "create 1" << std::endl;
    i = glthread.newRenderContextCall(2, window_id, 0);
    std::cout << "got render context id " << i << std::endl;
    sleep_for(1s);

    std::cout << "delete 1" << std::endl;
    glthread.delRenderContextCall(i);

    for (n = 0; n <= 8; n++)
    {
        std::cout << "create " << n + 2 << std::endl; // segfault at create 6
        i = glthread.newRenderContextCall(2, window_id, 0);
        std::cout << "got render context id " << i << std::endl;
        // del render group & context
        sleep_for(1s);
        glthread.delRenderContextCall(i); // no remove, no segfault
        // segfault occurs when uploading bitmaps into GPU memory
        std::cout << "delete " << n + 2 << std::endl;
    }

    // del render group
    glthread.delRenderGroupCall(window_id);

    std::cout << name << "stopping threads" << std::endl;
    livethread.stopCall();
    avthread.stopCall();
    glthread.stopCall();
}

void test_4()
{
    const char *name = "@TEST: live_av_openglthread_test2: test 4: ";
    std::cout << name << "** @@Play video from an sdp source **" << std::endl;
    int i;

    if (!stream_sdp)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_SDP" << std::endl;
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
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::sdp, std::string(stream_sdp), 2, &out_filter); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx);
    std::cout << name << "playing stream !" << std::endl;
    livethread.playStreamCall(ctx);

    // create render group & context
    glthread.newRenderGroupCall(window_id);
    i = glthread.newRenderContextCall(2, window_id, 0);
    std::cout << "got render context id " << i << std::endl;

    sleep_for(5s);
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

void test_5()
{

    const char *name = "@TEST: live_av_openglthread_test2: test 5: ";
    std::cout << name << "** @@DESCRIPTION **" << std::endl;
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
