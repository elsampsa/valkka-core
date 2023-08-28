/*
 * switch.cpp : Test the Switch and DoubleGate classes
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
 *  @file    switch.cpp
 *  @author  Sampsa Riikonen
 *  @date    2018
 *  @version 1.5.3 
 *  
 *  @brief   Test the Switch and DoubleGate classes
 *
 */ 

#include "framefifo.h"
#include "framefilter.h"
#include "framefilterset.h"
#include "logging.h"
#include "avdep.h"
#include "avthread.h"
#include "livethread.h"
#include "openglthread.h"
#include "test_import.h"


using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
  const char* name = "@TEST: switch: test 1: ";
  std::cout << name <<"** @@Stream from two cameras, switch between the two **" << std::endl;
  
  if (!stream_1) {
    std::cout << name <<"ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1"<< std::endl;
    exit(2);
  }
  std::cout << name <<"** test rtsp stream 1: "<< stream_1 << std::endl;
  if (!stream_2) {
    std::cout << name <<"ERROR: missing test stream 2: set environment variable VALKKA_TEST_RTSP_2"<< std::endl;
    exit(2);
  }
  std::cout << name <<"** test rtsp stream 2: "<< stream_1 << std::endl;
  
  /*
  filtergraph:                    +-------------------+       
                             +--> | channel_0         |
   (LiveThread:livethread)---|    |        Switch     +-----> {DummyFrameFilter:dummyfilter}
                             +--> | channel_1         |
                                  +-------------------+
  */
  
  DummyFrameFilter dummyfilter("dummy");
  Switch           sw("switch", &dummyfilter);
  LiveThread       livethread("livethread");
  std::cout << "starting live thread" << std::endl;
  
  livethread.startCall();
  
  LiveConnectionContext ctx  =LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 1, sw.getInputChannel(0));
  LiveConnectionContext ctx2 =LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_2), 2, sw.getInputChannel(1));
  
  livethread.registerStreamCall(ctx);
  livethread.registerStreamCall(ctx2);
  
  livethread.playStreamCall(ctx);
  livethread.playStreamCall(ctx2);
    
  sleep_for(5s);
  
  std::cout << "Enabling channel 0" << std::endl;
  sw.setChannel(0);
  sleep_for(5s);
  
  std::cout << "Enabling channel 1" << std::endl;
  sw.setChannel(1);
  sleep_for(5s);
  
  std::cout << "Enabling channel 0" << std::endl;
  sw.setChannel(0);
  sleep_for(5s);
  
  std::cout << "Enabling channel 1" << std::endl;
  sw.setChannel(1);
  sleep_for(5s);
  

  std::cout << "stopping live thread" << std::endl;
  livethread.stopCall();
}


void test_2() {
  const char* name = "@TEST: switch: test 2: ";
  std::cout << name <<"** @@Stream and decode from two cameras, switch between the two **" << std::endl;
  
  if (!stream_1) {
    std::cout << name <<"ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1"<< std::endl;
    exit(2);
  }
  std::cout << name <<"** test rtsp stream 1: "<< stream_1 << std::endl;
  if (!stream_2) {
    std::cout << name <<"ERROR: missing test stream 2: set environment variable VALKKA_TEST_RTSP_2"<< std::endl;
    exit(2);
  }
  std::cout << name <<"** test rtsp stream 2: "<< stream_1 << std::endl;
  
  /*
  filtergraph:                    +-------------------+       
                             +--> | channel_0         |
   (LiveThread:livethread)---|    |        Switch     +------>> (AVThread:avthread) -----> {DummyFrameFilter:dummyfilter}
                             +--> | channel_1         |
                                  +-------------------+
  */
  
  DummyFrameFilter dummyfilter("dummy");
  AVThread         avthread("avthread",dummyfilter);
  Switch           sw("switch", &(avthread.getFrameFilter()));
  LiveThread       livethread("livethread");
  
  std::cout << "starting threads" << std::endl;
  
  avthread.startCall();
  avthread.decodingOnCall();
  
  livethread.startCall();

  LiveConnectionContext ctx  =LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 1, sw.getInputChannel(0));
  LiveConnectionContext ctx2 =LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_2), 2, sw.getInputChannel(1));
  
  livethread.registerStreamCall(ctx);
  livethread.registerStreamCall(ctx2);
  
  livethread.playStreamCall(ctx);
  livethread.playStreamCall(ctx2);
    
  sleep_for(5s);
  
  std::cout << "Enabling channel 0" << std::endl;
  sw.setChannel(0);
  sleep_for(5s);
  
  std::cout << "Enabling channel 1" << std::endl;
  sw.setChannel(1);
  sleep_for(5s);
  
  std::cout << "Enabling channel 0" << std::endl;
  sw.setChannel(0);
  sleep_for(5s);
  
  std::cout << "Enabling channel 1" << std::endl;
  sw.setChannel(1);
  sleep_for(5s);
  
  std::cout << "stopping live thread" << std::endl;
  
  livethread.stopCall();
  avthread.stopCall();
}


void test_3() {
  const char* name = "@TEST: switch: test 3: ";
  std::cout << name <<"** @@Stream, decode and present from two cameras, switch between the two **" << std::endl;
  
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
  
  /*
  filtergraph:                                                   +-------------------+       
                             +--> {SlotFrameFilter: ch0} ------> | channel_0         |
   (LiveThread:livethread)---|                                   |        Switch     +------>> (AVThread:avthread) -----> {DummyFrameFilter:dummyfilter} ---->> (OpenGLThread:glthread)
                             +--> {SlotFrameFilter: ch1} ------> | channel_1         |
                                                                 +-------------------+
                                                                 
  The SlotFrameFilters change slot numbers for both streams coming from livethread to 1
  */
  
  OpenGLThread     glthread("gl_thread");
  DummyFrameFilter dummyfilter("dummy",true,&(glthread.getFrameFilter()));
  AVThread         avthread("avthread",dummyfilter);
  Switch           sw("switch", &(avthread.getFrameFilter()));
  
  // change both stream's slot number to 1
  SlotFrameFilter  ch0("ch0",1, sw.getInputChannel(0));
  SlotFrameFilter  ch1("ch1",1, sw.getInputChannel(1));
  
  LiveThread       livethread("livethread");
  
  std::cout << "starting threads" << std::endl;

  glthread.startCall();
  Window window_id=glthread.createWindow();
  glthread.makeCurrent(window_id);
  std::cout << "new x window "<<window_id<<std::endl;
  
  // create render group & context
  glthread.newRenderGroupCall(window_id);
  int i=glthread.newRenderContextCall(1, window_id, 0); // map slot 1 to window window_id
  std::cout << "got render context id "<<i<<std::endl;
  
  avthread.startCall();
  avthread.decodingOnCall();
  
  livethread.startCall();

  LiveConnectionContext ctx  =LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 1, &ch0);
  livethread.registerStreamCall(ctx);
  livethread.playStreamCall(ctx);

  LiveConnectionContext ctx2 =LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_2), 2, &ch1); // {SlotFrameFilter:ch1} changes the slot number from 2 => 1 
  livethread.registerStreamCall(ctx2);
  livethread.playStreamCall(ctx2);
  
  std::cout << "Enabling channel 0" << std::endl;
  sw.setChannel(1);
  sleep_for(5s);
  
  std::cout << "Enabling channel 1" << std::endl;
  sw.setChannel(0);
  sleep_for(5s);

  /*
  std::cout << "Enabling channel 0" << std::endl;
  sw.setChannel(0);
  sleep_for(5s);
  
  std::cout << "Enabling channel 1" << std::endl;
  sw.setChannel(1);
  sleep_for(5s);
  */
  
  // del render group & context
  glthread.delRenderContextCall(i);
  glthread.delRenderGroupCall(window_id);

  
  std::cout << "stopping threads" << std::endl;
  
  livethread.stopCall();
  avthread.stopCall();
  glthread.stopCall();
}


void test_4() {
  
  const char* name = "@TEST: switch: test 4: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}


void test_5() {
  
  const char* name = "@TEST: switch: test 5: ";
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


