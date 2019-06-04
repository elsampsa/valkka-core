/*
 * filethread_test.cpp : Test output with filethread
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
 *  @file    filethread_test.cpp
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
#include "test_import.h"


using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


// (FileThread:file) --> {InfoFrameFilter:out_filter} -->> (AVThread:avthread) --> {InfoFrameFilter:decoded_info} -->> (OpenGLThread:glthread)
OpenGLThread    glthread("gl_thread");
FifoFrameFilter &gl_in_filter = glthread.getFrameFilter();
InfoFrameFilter decoded_info("decoded",&gl_in_filter);
AVThread        avthread("avthread",decoded_info);
FifoFrameFilter &in_filter = avthread.getBlockingFrameFilter(); // request input framefilter from AVThread
InfoFrameFilter info_filter("encoded",&in_filter);
FileThread      filethread("filethread");


void test_1() {
  const char* name = "@TEST: file_test: test 1: ";
  std::cout << name <<"** @@Test FileThread **" << std::endl;
  
  std::cout << "starting file thread" << std::endl;
  filethread.startCall();
  
  sleep_for(2s);
  
  std::cout << "stopping file thread" << std::endl;
  filethread.stopCall();
}


void test_2() {
  const char* name = "@TEST: file_test: test 2: ";
  std::cout << name <<"** @@Stream from FileThread to AVThread and OpenGLThread. Seek. **" << std::endl;
  int i;
  long int duration, mstimestamp;
  
  std::cout << "starting threads" << std::endl;
  
  glthread.   startCall();
  filethread. startCall();
  avthread.   startCall();
  
  avthread.  decodingOnCall();
  
  Window window_id  =glthread.createWindow();
  glthread.makeCurrent(window_id);
  
  glthread.newRenderGroupCall(window_id);
  i =glthread.newRenderContextCall(1, window_id,  0);
  
  std::cout << "new x window "<<window_id<<std::endl;
  
  sleep_for(2s);
  
  std::cout << "registering stream" << std::endl;
  
  FileContext ctx=FileContext("kokkelis.mkv", 1, &info_filter); // , 0, &duration, &mstimestamp); // filename, slot, framefilter, start seek, etc.
  
  filethread.openFileStreamCall(ctx);
  
  std::cout << "got file status" << int(ctx.status) << std::endl;
  
  sleep_for(2s);
  
  /*
  std::cout << "seeking stream" << std::endl;
  
  ctx.seektime_=2000;
  filethread.seekFileStreamCall(ctx);
  */
  
  sleep_for(5s);
  
  std::cout << "stopping threads" << std::endl;
  
  // glthread.delRenderContextCall(i);
  // glthread.delRenderGroupCall(window_id); // TODO: what happends if you forget these..?  fix!  // TODO: what happens if file unavailable? fix!
  
  avthread.   stopCall();
  filethread. stopCall();
  glthread.   stopCall(); // TODO: print warning if you try to re-start a thread!
}


void test_3() {
  const char* name = "@TEST: file_test: test 3: ";
  std::cout << name <<"** @@Stream from FileThread to AVThread and OpenGLThread. Seek, play, stop, etc.**" << std::endl;
  int i;
  
  std::cout << "starting threads" << std::endl;
  glthread.   startCall();
  avthread.   startCall();
  filethread. startCall();
  
  avthread.  decodingOnCall();
  
  Window window_id  =glthread.createWindow();
  glthread.makeCurrent(window_id);
  
  glthread.newRenderGroupCall(window_id);
  i =glthread.newRenderContextCall(1, window_id,  0);
  
  std::cout << "new x window "<<window_id<<std::endl;
  
  // sleep_for(2s);
  
  std::cout << "registering stream" << std::endl;
  FileContext ctx=FileContext("kokkelis.mkv", 1, &info_filter); // 0, &duration, &mstimestamp); // filename, slot, framefilter, start seek, etc.
  filethread.openFileStreamCall(ctx);
  
  std::cout << "got file status" << int(ctx.status) << std::endl;
  
  // sleep_for(2s);

  std::cout << "\nPLAY\n";  
  filethread.playFileStreamCall(ctx);
  sleep_for(10s);
  
  std::cout << "\nSEEK\n";
  ctx.seektime_=2000;
  filethread.seekFileStreamCall(ctx);
  
  std::cout << "\nSTOP\n";
  filethread.stopFileStreamCall(ctx);
  sleep_for(3s);
  
  std::cout << "\nPLAY\n";  
  filethread.playFileStreamCall(ctx);
  sleep_for(5s);
  
  std::cout << "stopping threads" << std::endl;
  
  avthread.   stopCall();
  filethread. stopCall();
  // avthread.   stopCall();
  glthread.   stopCall();  
}


void test_4() {
  const char* name = "@TEST: file_test: test 4: ";
  std::cout << name <<"** @@Stream from FileThread to AVThread and OpenGLThread.  Two streams.  Seek, play, stop, etc.**" << std::endl;
  int i, i2;
  
  std::cout << "starting threads" << std::endl;
  glthread.   startCall();
  filethread. startCall();
  avthread.   startCall();
  
  avthread.  decodingOnCall();
  
  Window window_id  =glthread.createWindow();
  glthread.makeCurrent(window_id);
  
  Window window_id2 =glthread.createWindow();
  glthread.makeCurrent(window_id2);
  
  glthread.newRenderGroupCall(window_id);
  glthread.newRenderGroupCall(window_id2);
  
  i =glthread.newRenderContextCall(1, window_id,  0);
  i2=glthread.newRenderContextCall(2, window_id2, 0);
  
  std::cout << "new x window "<<window_id<<" "<<window_id2<<std::endl;
  std::cout << "registering stream" << std::endl;
  
  FileContext ctx  =FileContext("kokkelis.mkv", 1, &info_filter); // 0, &duration, &mstimestamp); // filename, slot, framefilter, start seek, etc.
  FileContext ctx2 =FileContext("kokkelis.mkv", 2, &info_filter); // 0, &duration, &mstimestamp); // filename, slot, framefilter, start seek, etc.
  
  filethread.openFileStreamCall(ctx);
  filethread.openFileStreamCall(ctx2);
  
  std::cout << "got file status  " << int(ctx.status) << std::endl;
  std::cout << "got file status2 " << int(ctx2.status) << std::endl;
  
  // sleep_for(2s);
  
  std::cout << "seeking stream" << std::endl;
  
  ctx.seektime_ =10000;
  ctx2.seektime_=10000;
  
  filethread.seekFileStreamCall(ctx);
  filethread.seekFileStreamCall(ctx2);
  
  sleep_for(3s);
  
  /*
  avthread.   stopCall();
  filethread.stopCall();
  glthread.   stopCall();
  
  return;
  */
  
  filethread.playFileStreamCall(ctx);
  filethread.playFileStreamCall(ctx2);
  
  sleep_for(5s);
  
  filethread.stopFileStreamCall(ctx);
  filethread.stopFileStreamCall(ctx2);
  
  sleep_for(5s);
  
  ctx.seektime_ =1000;
  ctx2.seektime_=1000;
  filethread.seekFileStreamCall(ctx);
  filethread.seekFileStreamCall(ctx2);
  
  sleep_for(3s);
  
  filethread.playFileStreamCall(ctx);
  filethread.playFileStreamCall(ctx2);
  
  sleep_for(3s);
  
  std::cout << "stopping threads" << std::endl;
  
  // glthread.delRenderContextCall(i);
  // glthread.delRenderGroupCall(window_id); // TODO: what happends if you forget these..?  fix!  // TODO: what happens if file unavailable? fix!
  
  avthread.   stopCall();
  filethread. stopCall();
  glthread.   stopCall(); // TODO: print warning if you try to re-start a thread!
  
}


void test_5() {
  const char* name = "@TEST: file_test: test 5: ";
  std::cout << name <<"** @@Stream from FileThread to AVThread and OpenGLThread. Seek and play. **" << std::endl;
  int i;
  
  
  std::cout << "starting threads" << std::endl;
  glthread.   startCall();
  filethread. startCall();
  avthread.   startCall();
  
  avthread.   decodingOnCall();
  
  Window window_id  =glthread.createWindow();
  glthread.makeCurrent(window_id);
  
  glthread.newRenderGroupCall(window_id);
  i =glthread.newRenderContextCall(1, window_id,  0);
  
  std::cout << "new x window "<<window_id<<std::endl;
  
  sleep_for(2s);
  
  std::cout << "registering stream" << std::endl;
  
  FileContext ctx=FileContext("kokkelis.mkv", 1, &info_filter); // , 0, &duration, &mstimestamp); // filename, slot, framefilter, start seek, etc.
  
  filethread.openFileStreamCall(ctx);
  std::cout << "got file status" << int(ctx.status) << std::endl;
  
  // sleep_for(2s);
  
  // case (1): immediate play
  filethread.playFileStreamCall(ctx);
  
  ///*
  // case (2): wait, seek, wait (1 sec), play
  sleep_for(5s);
  ctx.seektime_=0;
  std::cout << "\nSEEK\n\n";
  filethread.seekFileStreamCall(ctx);
  // sleep_for(1s);
  std::cout << "\nPLAY\n\n";
  filethread.playFileStreamCall(ctx);
  //*/
  
  // all cases: play for 5 secs
  sleep_for(5s);
  
  std::cout << "stopping threads" << std::endl;
  
  // glthread.delRenderContextCall(i);
  // glthread.delRenderGroupCall(window_id); // TODO: what happends if you forget these..?  fix!  // TODO: what happens if file unavailable? fix!
  
  avthread.   stopCall();
  filethread. stopCall();
  glthread.   stopCall(); // TODO: print warning if you try to re-start a thread!
}


void test_6() {
  const char* name = "@TEST: file_test: test 6: ";
  std::cout << name <<"** @@Stream from FileThread to AVThread. Seek, play, stop, etc.**" << std::endl;
  int i;
  
  std::cout << "starting threads" << std::endl;
  glthread.   startCall();
  avthread.   startCall();
  filethread. startCall();
  
  avthread.  decodingOnCall();
  
  Window window_id  =glthread.createWindow();
  glthread.makeCurrent(window_id);
  
  glthread.newRenderGroupCall(window_id);
  i =glthread.newRenderContextCall(1, window_id,  0);
  
  std::cout << "new x window "<<window_id<<std::endl;
  
  // sleep_for(2s);
  
  std::cout << "registering stream" << std::endl;
  FileContext ctx=FileContext("kokkelis.mkv", 1, &info_filter); // 0, &duration, &mstimestamp); // filename, slot, framefilter, start seek, etc.
  filethread.openFileStreamCall(ctx);
  
  std::cout << "got file status" << int(ctx.status) << std::endl;
  
  // sleep_for(2s);

  std::cout << "\nPLAY\n";  
  filethread.playFileStreamCall(ctx);
  sleep_for(10s);
  
  std::cout << "\nSEEK\n";
  ctx.seektime_=2000;
  filethread.seekFileStreamCall(ctx);
  
  std::cout << "\nSTOP\n";
  filethread.stopFileStreamCall(ctx);
  sleep_for(3s);
  
  std::cout << "\nPLAY\n";  
  filethread.playFileStreamCall(ctx);
  sleep_for(5s);
  
  std::cout << "stopping threads" << std::endl;
  
  filethread. stopCall();
  avthread.   stopCall();
  glthread.   stopCall();
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


