/*
 * file_test.cpp : Test file input
 * 
 * Copyright 2017-2023 Valkka Security Ltd. and Sampsa Riikonen
 * Copyright 2024 Sampsa Riikonen
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
 *  @file    file_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.6.1 
 *  
 *  @brief Test file input
 *
 */ 

#include "framefifo.h"
#include "framefilter.h"
#include "logging.h"
#include "avdep.h"
#include "livethread.h"
#include "fileframefilter.h"
#include "avfilethread.h"
#include "openglthread.h"
#include "test_import.h"


using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");

// filtergraph:
// (LiveThread:livethread) --> {FileFrameFilter:file_out}

// InfoFrameFilter     out_filter  ("out");
FileFrameFilter     file_out    ("file_writer");
// InfoFrameFilter info        ("info",&file_out);       
LiveThread      livethread  ("livethread"); 


void test_1() {
  const char* name = "@TEST: file_test: test 1: ";
  std::cout << name <<"** @@Test FileFrameFilter : produces kokkelis.mkv **" << std::endl;
  
  std::cout << name << "starting threads" << std::endl;
  livethread.startCall();
  
  sleep_for(2s);
  
  std::cout << name << "registering stream " << stream_1 << std::endl;
  // LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &info); // Request livethread to write into filter info
  LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &file_out); // Request livethread to write into filter info
  livethread.registerStreamCall(ctx);
  
  sleep_for(1s);
  
  std::cout << name << "playing stream !" << std::endl;
  livethread.playStreamCall(ctx);
  
  // sleep_for(2s);
  
  std::cout << name << "writing to disk" << std::endl;
  file_out.activate("kokkelis.mkv");
  
  sleep_for(5s);
  std::cout << name << "stop writing" << std::endl;
  
  file_out.deActivate();
  
  /*
  std::cout << name << "writing to disk - again" << std::endl;
  file_out.activate("kokkelis2.mkv");
  
  sleep_for(10s);
  std::cout << name << "stop writing" << std::endl;
  
  file_out.deActivate();
  */
  
  std::cout << name << "stopping threads" << std::endl;
  livethread.stopCall();  
}


void test_2() {
  const char* name = "@TEST: file_test: test 2: ";
  std::cout << name <<"** @@Test FileFrameFilter: wrong file **" << std::endl;
  
  std::cout << name << "starting threads" << std::endl;
  livethread.startCall();
  
  sleep_for(2s);
  
  std::cout << name << "registering stream" << std::endl;
  // LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &info); // Request livethread to write into filter info
  LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &file_out); // Request livethread to write into filter info
  livethread.registerStreamCall(ctx);
  
  sleep_for(1s);
  
  std::cout << name << "playing stream !" << std::endl;
  livethread.playStreamCall(ctx);
  
  // sleep_for(2s);
  
  std::cout << name << "writing to disk" << std::endl;
  file_out.activate("/root/kokkelis.mkv");
  
  sleep_for(5s);
  std::cout << name << "stop writing" << std::endl;
  
  file_out.deActivate();
  
  std::cout << name << "stopping threads" << std::endl;
  livethread.stopCall();  
}


void test_3() {
  const char* name = "@TEST: file_test: test 3: ";
  std::cout << name <<"** @@Test FileFrameFilter: activate, deActivate : produces kokkelis.mkv **" << std::endl;
  
  std::cout << name << "starting threads" << std::endl;
  livethread.startCall();
  
  sleep_for(2s);
  
  std::cout << name << "registering stream" << std::endl;
  // LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &info); // Request livethread to write into filter info
  LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &file_out); // Request livethread to write into filter info
  livethread.registerStreamCall(ctx);
  
  sleep_for(1s);
  
  std::cout << name << "playing stream !" << std::endl;
  livethread.playStreamCall(ctx);
  
  // sleep_for(2s);
  
  std::cout << name << "writing to disk" << std::endl;
  file_out.activate("kokkelis.mkv");
  
  sleep_for(5s);
  std::cout << name << "stop writing" << std::endl;
  
  file_out.deActivate();
  
  std::cout << name << "writing to disk - again" << std::endl;
  file_out.activate("kokkelis2.mkv");
  
  sleep_for(10s);
  std::cout << name << "stop writing" << std::endl;
  
  file_out.deActivate();
  
  std::cout << name << "stopping threads" << std::endl;
  livethread.stopCall();  
}


void test_4() {
  const char* name = "@TEST: file_test: test 4: ";
  std::cout << name <<"** @@Test FileFrameFilter: dirty close : produces kokkelis.mkv **" << std::endl;
  
  std::cout << name << "starting threads" << std::endl;
  livethread.startCall();
  
  sleep_for(2s);
  
  std::cout << name << "registering stream" << std::endl;
  // LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &info); // Request livethread to write into filter info
  LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &file_out); // Request livethread to write into filter info
  livethread.registerStreamCall(ctx);
  
  sleep_for(1s);
  
  std::cout << name << "playing stream !" << std::endl;
  livethread.playStreamCall(ctx);
  
  // sleep_for(2s);
  
  std::cout << name << "writing to disk" << std::endl;
  file_out.activate("kokkelis.mkv");
  
  sleep_for(5s);
  
  std::cout << name << "stopping threads" << std::endl;
  livethread.stopCall();  
}


void test_5() {
  
  const char* name = "@TEST: NAME: test 5: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}



int main(int argc, char** argcv) {
  if (argc<2) {
    std::cout << argcv[0] << " needs an integer argument.  Second interger argument (optional) is verbosity" << std::endl;
  }
  else {
    ffmpeg_av_register_all(); // never forget!
  
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


