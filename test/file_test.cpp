/*
 * file_test.cpp : Test file input/output
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
 *  @file    file_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.0 
 *  
 *  @brief Test file input/output
 *
 */ 

#include "queues.h"
#include "livethread.h"
#include "filethread.h"
#include "avthread.h"
#include "openglthread.h"
#include "filters.h"
#include "logging.h"
#include "file.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
  const char* name = "@TEST: file_test: test 1: ";
  std::cout << name <<"** @@Test FileFrameFilter **" << std::endl;
  
  // *************
  // filtergraph:
  // (LiveThread:livethread) --> {FileFrameFilter:file_out}
  
  // InfoFrameFilter     out_filter  ("out");
  FileFrameFilter     file_out    ("file_writer");
  // InfoFrameFilter info        ("info",&file_out);       
  LiveThread      livethread  ("livethread"); 
  // *************
  
  bool verbose;
  
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
  
  sleep_for(2s);
  
  std::cout << name << "writing to disk" << std::endl;
  file_out.activate("kokkelis.mkv");
  
  sleep_for(60s);
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


void test_2() {
  const char* name = "@TEST: file_test: test 2: ";
  std::cout << name <<"** @@Test FileThread **" << std::endl;
  
  FileThread file_thread("file_thread");
  
  std::cout << "starting file thread" << std::endl;
  
  file_thread.startCall();
  
  sleep_for(2s);
  
  std::cout << "stopping file thread" << std::endl;
  
  file_thread.stopCall();
  
}


void test_3() {
  const char* name = "@TEST: file_test: test 3: ";
  std::cout << name <<"** @@Test FileThread **" << std::endl;
  
  FileThread      file_thread("file_thread");
  InfoFrameFilter info        ("info");
  // long int        duration, mstimestamp;
  
  std::cout << "starting file thread" << std::endl;
  
  file_thread.startCall();
  
  sleep_for(2s);
  
  std::cout << "registering stream" << std::endl;
  
  FileContext ctx=FileContext("kokkelis.mkv", 1, &info); // 0, &duration, &mstimestamp); // filename, slot, framefilter, start seek, etc.
  
  file_thread.openFileStreamCall(ctx);
  
  sleep_for(2s);
  
  std::cout << "seeking stream" << std::endl;
  
  ctx.seektime_=2000;
  file_thread.seekFileStreamCall(ctx);
  
  sleep_for(5s);
  
  std::cout << "stopping file thread" << std::endl;
  
  file_thread.stopCall();
}


void test_4() {
  const char* name = "@TEST: file_test: test 4: ";
  std::cout << name <<"** @@Stream from FileThread to AVThread and OpenGLThread. Seek. **" << std::endl;
  int i;
  
  // (LiveThread:livethread) --> {FifoFrameFilter:av_in_filter} --> [FrameFifo:av_fifo] -->> (AVThread:avthread) --> {OpenGLFrameFifo:gl_in_filter} --> [OpenGLFrameFifo:gl_fifo] -->> (OpenGLThread:glthread)
  
  OpenGLThread      glthread        ("glthread",/*n720p*/10,/*n1080p*/10,/*n1440p*/0,/*4K*/0,/*msbuftime*/500,/*core_id*/-1); // remember buffering time!
  OpenGLFrameFifo&  gl_fifo         =glthread.getFifo();      // get gl_fifo from glthread
  FifoFrameFilter   gl_in_filter    ("gl_in_filter",gl_fifo);  
  
  // InfoFrameFilter   info            ("info");
  
  FrameFifo                 av_fifo         ("av_fifo",10); // TODO: we need here a fifo that waits ..
  AVThread                  avthread        ("avthread",av_fifo,gl_in_filter);  // [av_fifo] -->> (avthread) --> {gl_in_filter}
  BlockingFifoFrameFilter   av_in_filter    ("av_in_filter",av_fifo); // TODO: better idea .. FrameFifo could have two writing methods: blocking and non-blocking .. it depends on the filter which one it calls
  
  FileThread      file_thread("file_thread");
  long int        duration, mstimestamp;
  
  std::cout << "starting threads" << std::endl;
  
  glthread.   startCall();
  file_thread.startCall();
  avthread.   startCall();
  
  avthread.  decodingOnCall();
  
  Window window_id  =glthread.createWindow();
  glthread.makeCurrent(window_id);
  
  glthread.newRenderGroupCall(window_id);
  i =glthread.newRenderContextCall(1, window_id,  0);
  
  std::cout << "new x window "<<window_id<<std::endl;
  
  sleep_for(2s);
  
  std::cout << "registering stream" << std::endl;
  
  FileContext ctx=FileContext("kokkelis.mkv", 1, &av_in_filter); // , 0, &duration, &mstimestamp); // filename, slot, framefilter, start seek, etc.
  
  file_thread.openFileStreamCall(ctx);
  
  std::cout << "got file status" << int(ctx.status) << std::endl;
  
  sleep_for(2s);
  
  /*
  std::cout << "seeking stream" << std::endl;
  
  ctx.seektime_=2000;
  file_thread.seekFileStreamCall(ctx);
  */
  
  sleep_for(5s);
  
  std::cout << "stopping threads" << std::endl;
  
  // glthread.delRenderContextCall(i);
  // glthread.delRenderGroupCall(window_id); // TODO: what happends if you forget these..?  fix!  // TODO: what happens if file unavailable? fix!
  
  avthread.   stopCall();
  file_thread.stopCall();
  glthread.   stopCall(); // TODO: print warning if you try to re-start a thread!
}


void test_5() {
  const char* name = "@TEST: file_test: test 5: ";
  std::cout << name <<"** @@Stream from FileThread to AVThread and OpenGLThread. Seek, play, stop, etc.**" << std::endl;
  int i;
  
  // (LiveThread:livethread) --> {FifoFrameFilter:av_in_filter} --> [FrameFifo:av_fifo] -->> (AVThread:avthread) --> {OpenGLFrameFifo:gl_in_filter} --> [OpenGLFrameFifo:gl_fifo] -->> (OpenGLThread:glthread)
  
  OpenGLThread      glthread        ("glthread",/*n720p*/10,/*n1080p*/10,/*n1440p*/0,/*4K*/0,/*msbuftime*/500,/*core_id*/-1); // remember buffering time!
  OpenGLFrameFifo&  gl_fifo         =glthread.getFifo();      // get gl_fifo from glthread
  FifoFrameFilter   gl_in_filter    ("gl_in_filter",gl_fifo);  
  
  // InfoFrameFilter   info            ("info");
  
  FrameFifo                 av_fifo         ("av_fifo",10); // TODO: we need here a fifo that waits ..
  AVThread                  avthread        ("avthread",av_fifo,gl_in_filter);  // [av_fifo] -->> (avthread) --> {gl_in_filter}
  BlockingFifoFrameFilter   av_in_filter    ("av_in_filter",av_fifo); // TODO: better idea .. FrameFifo could have two writing methods: blocking and non-blocking .. it depends on the filter which one it calls
  
  FileThread      file_thread("file_thread");
  long int        duration, mstimestamp;
  
  std::cout << "starting threads" << std::endl;
  
  glthread.   startCall();
  file_thread.startCall();
  avthread.   startCall();
  
  avthread.  decodingOnCall();
  
  Window window_id  =glthread.createWindow();
  glthread.makeCurrent(window_id);
  
  glthread.newRenderGroupCall(window_id);
  i =glthread.newRenderContextCall(1, window_id,  0);
  
  std::cout << "new x window "<<window_id<<std::endl;
  
  // sleep_for(2s);
  
  std::cout << "registering stream" << std::endl;
  
  FileContext ctx=FileContext("kokkelis.mkv", 1, &av_in_filter); // 0, &duration, &mstimestamp); // filename, slot, framefilter, start seek, etc.
  
  file_thread.openFileStreamCall(ctx);
  
  std::cout << "got file status" << int(ctx.status) << std::endl;
  
  // sleep_for(2s);
  
  std::cout << "seeking stream" << std::endl;
  
  // ctx.seektime_=2000;
  ctx.seektime_=10000;
  // ctx.seektime_=20000;
  // ctx.seektime_=30000;
  
  file_thread.seekFileStreamCall(ctx);
  
  sleep_for(3s);
  std::cout << "\nPLAY\n";
  
  file_thread.playFileStreamCall(ctx);
  
  sleep_for(5s);
  std::cout << "\nSTOP\n";
  
  file_thread.stopFileStreamCall(ctx);
  
  sleep_for(5s);
  std::cout << "\nSEEK\n";
  
  ctx.seektime_=1000;
  file_thread.seekFileStreamCall(ctx);
  
  sleep_for(3s);
  std::cout << "\nPLAY\n";
  
  file_thread.playFileStreamCall(ctx);
  
  sleep_for(3s);
  std::cout << "stopping threads" << std::endl;
  
  // glthread.delRenderContextCall(i);
  // glthread.delRenderGroupCall(window_id); // TODO: what happends if you forget these..?  fix!  // TODO: what happens if file unavailable? fix!
  
  avthread.   stopCall();
  file_thread.stopCall();
  glthread.   stopCall(); // TODO: print warning if you try to re-start a thread!
  
}


void test_6() {
  const char* name = "@TEST: file_test: test 6: ";
  std::cout << name <<"** @@Stream from FileThread to AVThread and OpenGLThread.  Two stream.  Seek, play, stop, etc.**" << std::endl;
  int i, i2;
  
  // (LiveThread:livethread) --> {FifoFrameFilter:av_in_filter} --> [FrameFifo:av_fifo] -->> (AVThread:avthread) --> {OpenGLFrameFifo:gl_in_filter} --> [OpenGLFrameFifo:gl_fifo] -->> (OpenGLThread:glthread)
  
  OpenGLThread      glthread        ("glthread",/*n720p*/10,/*n1080p*/10,/*n1440p*/0,/*4K*/0,/*msbuftime*/500,/*core_id*/-1); // remember buffering time!
  OpenGLFrameFifo&  gl_fifo         =glthread.getFifo();      // get gl_fifo from glthread
  FifoFrameFilter   gl_in_filter    ("gl_in_filter",gl_fifo);  
  
  // InfoFrameFilter   info            ("info");
  
  FrameFifo                 av_fifo         ("av_fifo",10); // TODO: we need here a fifo that waits ..
  AVThread                  avthread        ("avthread",av_fifo,gl_in_filter);  // [av_fifo] -->> (avthread) --> {gl_in_filter}
  BlockingFifoFrameFilter   av_in_filter    ("av_in_filter",av_fifo); // TODO: better idea .. FrameFifo could have two writing methods: blocking and non-blocking .. it depends on the filter which one it calls
  
  FileThread      file_thread("file_thread");
  long int        duration, mstimestamp;
  
  std::cout << "starting threads" << std::endl;
  
  glthread.   startCall();
  file_thread.startCall();
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
  
  FileContext ctx  =FileContext("kokkelis.mkv", 1, &av_in_filter); // 0, &duration, &mstimestamp); // filename, slot, framefilter, start seek, etc.
  FileContext ctx2 =FileContext("kokkelis.mkv", 2, &av_in_filter); // 0, &duration, &mstimestamp); // filename, slot, framefilter, start seek, etc.
  
  file_thread.openFileStreamCall(ctx);
  file_thread.openFileStreamCall(ctx2);
  
  std::cout << "got file status  " << int(ctx.status) << std::endl;
  std::cout << "got file status2 " << int(ctx2.status) << std::endl;
  
  // sleep_for(2s);
  
  std::cout << "seeking stream" << std::endl;
  
  ctx.seektime_ =10000;
  ctx2.seektime_=10000;
  
  file_thread.seekFileStreamCall(ctx);
  file_thread.seekFileStreamCall(ctx2);
  
  sleep_for(3s);
  
  /*
  avthread.   stopCall();
  file_thread.stopCall();
  glthread.   stopCall();
  
  return;
  */
  
  file_thread.playFileStreamCall(ctx);
  file_thread.playFileStreamCall(ctx2);
  
  sleep_for(5s);
  
  file_thread.stopFileStreamCall(ctx);
  file_thread.stopFileStreamCall(ctx2);
  
  sleep_for(5s);
  
  ctx.seektime_ =1000;
  ctx2.seektime_=1000;
  file_thread.seekFileStreamCall(ctx);
  file_thread.seekFileStreamCall(ctx2);
  
  sleep_for(3s);
  
  file_thread.playFileStreamCall(ctx);
  file_thread.playFileStreamCall(ctx2);
  
  sleep_for(3s);
  
  std::cout << "stopping threads" << std::endl;
  
  // glthread.delRenderContextCall(i);
  // glthread.delRenderGroupCall(window_id); // TODO: what happends if you forget these..?  fix!  // TODO: what happens if file unavailable? fix!
  
  avthread.   stopCall();
  file_thread.stopCall();
  glthread.   stopCall(); // TODO: print warning if you try to re-start a thread!
  
}


void test_7() {
  const char* name = "@TEST: file_test: test 7: ";
  std::cout << name <<"** @@Stream from FileThread to AVThread and OpenGLThread. Seek and play. **" << std::endl;
  int i;
  
  // (LiveThread:livethread) --> {FifoFrameFilter:av_in_filter} --> [FrameFifo:av_fifo] -->> (AVThread:avthread) --> {OpenGLFrameFifo:gl_in_filter} --> [OpenGLFrameFifo:gl_fifo] -->> (OpenGLThread:glthread)
  
  OpenGLThread      glthread        ("glthread",/*n720p*/10,/*n1080p*/10,/*n1440p*/0,/*4K*/0,/*msbuftime*/500,/*core_id*/-1); // remember buffering time!
  OpenGLFrameFifo&  gl_fifo         =glthread.getFifo();      // get gl_fifo from glthread
  FifoFrameFilter   gl_in_filter    ("gl_in_filter",gl_fifo);  
  
  // InfoFrameFilter   info            ("info");
  
  FrameFifo                 av_fifo         ("av_fifo",10); // TODO: we need here a fifo that waits ..
  AVThread                  avthread        ("avthread",av_fifo,gl_in_filter);  // [av_fifo] -->> (avthread) --> {gl_in_filter}
  BlockingFifoFrameFilter   av_in_filter    ("av_in_filter",av_fifo); // TODO: better idea .. FrameFifo could have two writing methods: blocking and non-blocking .. it depends on the filter which one it calls
  
  FileThread      file_thread("file_thread");
  long int        duration, mstimestamp;
  
  std::cout << "starting threads" << std::endl;
  
  glthread.   startCall();
  file_thread.startCall();
  avthread.   startCall();
  
  avthread.  decodingOnCall();
  
  Window window_id  =glthread.createWindow();
  glthread.makeCurrent(window_id);
  
  glthread.newRenderGroupCall(window_id);
  i =glthread.newRenderContextCall(1, window_id,  0);
  
  std::cout << "new x window "<<window_id<<std::endl;
  
  sleep_for(2s);
  
  std::cout << "registering stream" << std::endl;
  
  FileContext ctx=FileContext("kokkelis.mkv", 1, &av_in_filter); // , 0, &duration, &mstimestamp); // filename, slot, framefilter, start seek, etc.
  
  file_thread.openFileStreamCall(ctx);
  std::cout << "got file status" << int(ctx.status) << std::endl;
  
  // sleep_for(2s);
  
  // case (1): immediate play
  file_thread.playFileStreamCall(ctx);
  
  ///*
  // case (2): wait, seek, wait (1 sec), play
  sleep_for(5s);
  ctx.seektime_=0;
  std::cout << "\nSEEK\n\n";
  file_thread.seekFileStreamCall(ctx);
  // sleep_for(1s);
  std::cout << "\nPLAY\n\n";
  file_thread.playFileStreamCall(ctx);
  //*/
  
  // all cases: play for 5 secs
  sleep_for(5s);
  
  std::cout << "stopping threads" << std::endl;
  
  // glthread.delRenderContextCall(i);
  // glthread.delRenderGroupCall(window_id); // TODO: what happends if you forget these..?  fix!  // TODO: what happens if file unavailable? fix!
  
  avthread.   stopCall();
  file_thread.stopCall();
  glthread.   stopCall(); // TODO: print warning if you try to re-start a thread!
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
      case(6):
        test_6();
        break; 
      case(7):
        test_7();
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


