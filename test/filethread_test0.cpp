/*
 * filethread_test0.cpp : Test output with filethread.  No frame visualization with OpenGL
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
 *  @file    filethread_test0.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.11.0 
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
#include "avthread.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


// (FileThread:file) --> {InfoFrameFilter:out_filter} -->> (AVThread:avthread) --> {InfoFrameFilter:decoded_info} -->> (OpenGLThread:glthread)
///*
DummyFrameFilter decoded_info("decoded",false);
AVThread        avthread("avthread",decoded_info);

FifoFrameFilter &in_filter = avthread.getBlockingFrameFilter(); // request input framefilter from AVThread // TODO: using this sucker blocks the thread!
// FifoFrameFilter &in_filter = avthread.getFrameFilter(); // request input framefilter from AVThread

InfoFrameFilter info_filter("encoded",&in_filter);
//*/

// DummyFrameFilter info_filter("encoded",false);

FileThread      filethread("filethread");


void test_1() {
  const char* name = "@TEST: file_test: test 1: ";
  std::cout << name <<"** @@Stream from FileThread to AVThread. Seek, play, stop, etc.**" << std::endl;
  int i;
  
  std::cout << "starting threads" << std::endl;
  avthread.   startCall();
  filethread. startCall();
  
  avthread.  decodingOnCall();
  
  // sleep_for(2s);
  
  std::cout << "registering stream" << std::endl;
  FileContext ctx=FileContext("kokkelis.mkv", 1, &info_filter); // 0, &duration, &mstimestamp); // filename, slot, framefilter, start seek, etc.
  filethread.openFileStreamCall(ctx);
  
  std::cout << "got file status" << int(ctx.status) << std::endl;
  
  // sleep_for(2s);

  std::cout << "\nPLAY\n";  
  filethread.playFileStreamCall(ctx);
  sleep_for(10s);
  
  /*
  std::cout << "\nSEEK\n";
  ctx.seektime_=2000;
  filethread.seekFileStreamCall(ctx);
  
  std::cout << "\nSTOP\n";
  filethread.stopFileStreamCall(ctx);
  sleep_for(3s);
  
  std::cout << "\nPLAY\n";  
  filethread.playFileStreamCall(ctx);
  sleep_for(5s);
  */
  
  std::cout << "stopping threads" << std::endl;
  
  filethread. stopCall();
  avthread.   stopCall();  
}


void test_2() {
  const char* name = "@TEST: file_test: test 2: ";
  std::cout << name <<"** @@Pull all packets from a file using TestFileStream**" << std::endl;
  int i;
  
  TestFileStream stream =TestFileStream("kokkelis.mkv");
  
  stream.pull();
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
      /*
      case(3):
        test_3();
        break;
      case(4):
        test_4();
        break;
      case(5):
        test_5();
        break;
      */
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


