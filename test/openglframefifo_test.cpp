/*
 * openglframefifo_test.cpp :
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
 *  @file    openglframefifo_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2018
 *  @version 0.4.4 
 *  
 *  @brief 
 *
 */ 

#include "framefifo.h"
#include "framefilter.h"
#include "logging.h"
#include "avdep.h"
#include "openglframefifo.h"
#include "avthread.h"
#include "livethread.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");

// (LiveThread:livethread) --> {FrameFilter:info} --> {FifoFrameFilter:in_filter} -->> (AVThread:avthread) --> {FifoFrameFilter:gl_in_filter} --> [OpenGLFrameFifo]

// OpenGLFrameFifo* gl_fifo = new OpenGLFrameFifo();
OpenGLFrameFifo* gl_fifo = new OpenGLFrameFifo();
FifoFrameFilter  gl_in_filter("gl_in_filter",gl_fifo);
InfoFrameFilter  decoded_info("decoded",&gl_in_filter);
AVThread         avthread("avthread",decoded_info);
FifoFrameFilter  &in_filter = avthread.getFrameFilter(); // request framefilter from AVThread
InfoFrameFilter  out_filter("encoded",&in_filter);
LiveThread       livethread("live");


void test_1() {
  
  const char* name = "@TEST: openglframefifo_test: test 1: live => decoder => openglframefifo ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
  if (!stream_1) {
    std::cout << name <<"ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1"<< std::endl;
    exit(2);
  }
  std::cout << name <<"** test rtsp stream 1: "<< stream_1 << std::endl;
  
  std::cout << name << "starting threads" << std::endl;
  
  // gl_fifo->debugOn();
  gl_fifo->allocateYUV(); // normally done by OpenGLThread
  
  livethread.startCall();
  avthread.  startCall();

  avthread.decodingOnCall();
  
  sleep_for(2s);
  
  std::cout << name << "registering stream" << std::endl;
  LiveConnectionContext ctx =LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &out_filter); // Request livethread to write into filter info
  livethread.registerStreamCall(ctx);
  
  sleep_for(1s);
  
  std::cout << name << "playing stream !" << std::endl;
  livethread.playStreamCall(ctx);
  
  sleep_for(10s);
  // sleep_for(604800s); //one week
  
  std::cout << name << "stopping threads" << std::endl;
  livethread.stopCall();
  avthread.  stopCall();
  // AVThread destructor => Thread destructor => stopCall (Thread::stopCall or AVThread::stopCall ..?) .. in destructor, virtual methods are not called
  // 
  
  gl_fifo->deallocateYUV(); // normally done by OpenGLThread
  
  // the correct order here: stop livethread, stop avthread, finally, deallocateYUV (otherwise avthread is using deallocated yuv frames)
}


void test_2() {
  
  const char* name = "@TEST: openglframefifo_test: test 2: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}


void test_3() {
  
  const char* name = "@TEST: openglframefifo_test: test 3: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}


void test_4() {
  
  const char* name = "@TEST: openglframefifo_test: test 4: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}


void test_5() {
  
  const char* name = "@TEST: openglframefifo_test: test 5: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}



int main(int argc, char** argcv) {
  if (argc<2) {
    std::cout << argcv[0] << " needs an integer argument.  Second interger argument (optional) is verbosity" << std::endl;
  }
  else {
    
    if  (argc>2) { // choose verbosity
      switch (atoi(argcv[2])) {
        case(0): // shut up
          ffmpeg_av_log_set_level(0);
          fatal_log_all();
          break;
        case(1): // normal
          break;
        case(2): // more verbose
          ffmpeg_av_log_set_level(100);
          debug_log_all();
          break;
        case(3): // extremely verbose
          ffmpeg_av_log_set_level(100);
          crazy_log_all();
          break;
        default:
          std::cout << "Unknown verbosity level "<< atoi(argcv[2]) <<std::endl;
          exit(1);
          break;
      }
    }
    
    switch (atoi(argcv[1])) { // choose test
      case(1):
        test_1();
        break;
      case(2):
        test_2();
        break;
      case(3):
        test_3();
        break;
      case(4):
        test_4();
        break;
      case(5):
        test_5();
        break;
      default:
        std::cout << "No such test "<<argcv[1]<<" for "<<argcv[0]<<std::endl;
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


