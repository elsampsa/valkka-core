/*
 * opengl_thread_test.cpp : Test OpenGLThread signals and functionality.  Test whole pipeline from LiveThread to OpenGL rendering.
 * 
 * Copyright 2017 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Valkka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Valkka.  If not, see <http://www.gnu.org/licenses/>. 
 * 
 */

/** 
 *  @file    opengl_thread_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
 *  
 *  @brief Test OpenGLThread signals and functionality. Test whole pipeline from LiveThread to OpenGL rendering
 */ 

#include "queues.h"
#include "livethread.h"
#include "avthread.h"
#include "logging.h"
#include "openglthread.h"
#include "filters.h"
#include "chains.h"
#include "logging.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
  const char* name = "@TEST: opengl_thread_test: test 1: ";
  std::cout << name <<"** @@OpenGLThread signals **" << std::endl;
  
  int i;
  OpenGLThread glthread("glthread",/*n720p*/10,/*n1080p*/0,/*n1440p*/0,/*4K*/0,/*naudio*/0,/*msbuftime*/0,/*core_id*/-1);
  
  glthread.startCall();
  
  Window window_id=glthread.createWindow();
  glthread.makeCurrent(window_id);
  std::cout << "new x window "<<window_id<<std::endl;
  
  // (1)
  glthread.newRenderGroupCall(window_id);
  glthread.infoCall();
  sleep_for(1s);
  
  i=glthread.newRenderContextCall(1, window_id, 0);
  std::cout << "got render context id "<<i<<std::endl;
  glthread.infoCall();
  sleep_for(1s);
  
  glthread.delRenderContextCall(i);
  glthread.infoCall();
  sleep_for(1s);
  
  glthread.delRenderGroupCall(window_id);
  glthread.infoCall();
  sleep_for(1s);
  
  std::cout<<std::endl<<std::endl<<std::endl;
  
  // (2)
  glthread.newRenderGroupCall(window_id);
  glthread.infoCall();
  sleep_for(1s);
  
  i=glthread.newRenderContextCall(1, window_id, 0);
  std::cout << "got render context id "<<i<<std::endl;
  glthread.infoCall();
  sleep_for(1s);
  
  glthread.delRenderGroupCall(window_id);
  glthread.infoCall();
  sleep_for(1s);
  
  glthread.stopCall();
}


void test_2() {
  const char* name = "@TEST: opengl_thread_test: test 2: ";
  std::cout << name <<"** @@OpenGLThread live rendering **" << std::endl;
  
  if (!stream_1) {
    std::cout << name <<"ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1"<< std::endl;
    exit(2);
  }
  std::cout << name <<"** test rtsp stream 1: "<< stream_1 << std::endl;
  
  int i;
  // ***********************************
  // filtergraph:
  // (LiveThread:livethread) --> {InfoFrameFilter:live_out_filter} --> {FifoFrameFilter:av_in_filter} --> [FrameFifo:av_fifo] -->> (AVThread:avthread) --> {FifoFrameFilter:gl_in_gilter} --> 
  // --> [OpenGLFrameFifo:gl_fifo] -->> (OpenGLThread:glthread)
  //
  OpenGLThread      glthread        ("glthread",/*n720p*/10,/*n1080p*/10,/*n1440p*/0,/*4K*/0,/*naudio*/10,/*msbuftime*/100,/*core_id*/-1); // remember buffering time!
  OpenGLFrameFifo&  gl_fifo         =glthread.getFifo();      // get gl_fifo from glthread
  FifoFrameFilter   gl_in_filter    ("gl_in_filter",gl_fifo);   
  
  FrameFifo         av_fifo         ("av_fifo",10);                 
  AVThread          avthread        ("avthread",av_fifo,gl_in_filter);  // [av_fifo] -->> (avthread) --> {gl_in_filter}
  
  FifoFrameFilter   av_in_filter    ("av_in_filter",av_fifo);
  // InfoFrameFilter   live_out_filter ("live_out_filter",&av_in_filter);
  // DummyFrameFilter   live_out_filter ("live_out_filter",false,&av_in_filter);
  BriefInfoFrameFilter   live_out_filter ("live_out_filter",&av_in_filter);
  LiveThread        livethread      ("livethread");
  // ***********************************
  
  LiveConnectionContext ctx;
  
  std::cout << name << "starting threads" << std::endl;
  glthread.startCall(); // start running OpenGLThread!
  
  Window window_id=glthread.createWindow();
  glthread.makeCurrent(window_id);
  std::cout << "new x window "<<window_id<<std::endl;
  
  livethread.startCall();
  avthread.  startCall();

  avthread.decodingOnCall();
  
  // sleep_for(1s);
  
  std::cout << name << "registering stream" << std::endl;
  ctx = (LiveConnectionContext){LiveConnectionType::rtsp, std::string(stream_1), 1, &live_out_filter}; // Request livethread to write into filter info
  livethread.registerStreamCall(ctx);
  
  // sleep_for(1s);
  std::cout << name << "playing stream !" << std::endl;
  livethread.playStreamCall(ctx);
  
  // (1)
  glthread.newRenderGroupCall(window_id);
  sleep_for(1s);
  
  i=glthread.newRenderContextCall(1, window_id, 0);
  std::cout << "got render context id "<<i<<std::endl;
  sleep_for(1s);
  
  // sleep_for(3s);
  sleep_for(10s);
  
  glthread.delRenderContextCall(i);
  glthread.delRenderGroupCall(window_id);
  
  std::cout << name << "stopping threads" << std::endl;
  livethread.stopCall();
  avthread.  stopCall();
  glthread.  stopCall();
  std::cout << name << "All threads stopped" << std::endl;
  sleep_for(1s);
  std::cout << name << "Leaving context" << std::endl;
}


void test_3() {
  const char* name = "@TEST: opengl_thread_test: test 3: ";
  std::cout << name <<"** @@OpenGL live rendering with two rtsp cams **" << std::endl;
  
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
  
  int i, i2;
  // ***********************************
  // filtergraph:
  // (LiveThread:livethread) --> {InfoFrameFilter:live_out_filter} --> {FifoFrameFilter:av_in_filter} --> [FrameFifo:av_fifo] -->> (AVThread:avthread) --> {FifoFrameFilter:gl_in_gilter} --> 
  // --> [OpenGLFrameFifo:gl_fifo] -->> (OpenGLThread:glthread)
  //
  OpenGLThread      glthread        ("glthread",/*n720p*/10,/*n1080p*/10,/*n1440p*/0,/*4K*/0,/*naudio*/10,/*msbuftime*/100,/*core_id*/-1); // remember buffering time!
  OpenGLFrameFifo&  gl_fifo         =glthread.getFifo();      // get gl_fifo from glthread
  FifoFrameFilter   gl_in_filter    ("gl_in_filter",gl_fifo);   
  
  // first stream
  FrameFifo         av_fifo         ("av_fifo",10);                 
  AVThread          avthread        ("avthread",av_fifo,gl_in_filter);  // [av_fifo] -->> (avthread) --> {gl_in_filter}
  FifoFrameFilter   av_in_filter    ("av_in_filter",av_fifo);
  InfoFrameFilter   live_out_filter ("live_out_filter",&av_in_filter);
  
  // second stream
  FrameFifo         av_fifo2        ("av_fifo2",10);                 
  AVThread          avthread2       ("avthread2",av_fifo2,gl_in_filter);  // [av_fifo] -->> (avthread) --> {gl_in_filter}
  FifoFrameFilter   av_in_filter2   ("av_in_filter2",av_fifo2);
  InfoFrameFilter   live_out_filter2("live_out_filter2",&av_in_filter2);
  
  LiveThread        livethread      ("livethread");
  // ***********************************
  
  LiveConnectionContext ctx;
  LiveConnectionContext ctx2;
  
  std::cout << name << "starting threads" << std::endl;
  glthread.startCall(); // start running OpenGLThread!
  
  Window window_id  =glthread.createWindow();
  glthread.makeCurrent(window_id);
  std::cout << "new x window "<<window_id<<std::endl;
  
  Window window_id2 =glthread.createWindow();
  glthread.makeCurrent(window_id2);
  std::cout << "new x window "<<window_id2<<std::endl;
  
  livethread.startCall();
  
  avthread.  startCall();
  avthread2. startCall();
 
  avthread.  decodingOnCall();
  avthread2. decodingOnCall();
  
  // sleep_for(1s);
  
  std::cout << name << "registering stream" << std::endl;
  
  ctx = (LiveConnectionContext){LiveConnectionType::rtsp, std::string(stream_1), 1, &live_out_filter}; // Request livethread to write into filter info
  ctx2= (LiveConnectionContext){LiveConnectionType::rtsp, std::string(stream_2), 2, &live_out_filter2}; // Request livethread to write into filter info
  
  livethread.registerStreamCall(ctx);
  livethread.registerStreamCall(ctx2);
  
  // sleep_for(1s);
  std::cout << name << "playing stream !" << std::endl;
  livethread.playStreamCall(ctx);
  livethread.playStreamCall(ctx2);
  
  glthread.newRenderGroupCall(window_id);
  glthread.newRenderGroupCall(window_id2);
  sleep_for(1s);
  
  i =glthread.newRenderContextCall(1, window_id,  0);
  i2=glthread.newRenderContextCall(2, window_id2, 0);
  std::cout << "got render context id "<<i<<std::endl;
  sleep_for(1s);
  
  // sleep_for(3s);
  sleep_for(10s);
  
  glthread.delRenderContextCall(i);
  glthread.delRenderContextCall(i2);
  
  glthread.delRenderGroupCall(window_id);
  glthread.delRenderGroupCall(window_id2);
  
  std::cout << name << "stopping threads" << std::endl;
  livethread.stopCall();
  
  avthread.  stopCall();
  avthread2. stopCall();
  
  glthread.  stopCall();
  std::cout << name << "All threads stopped" << std::endl;
  sleep_for(1s);
  std::cout << name << "Leaving context" << std::endl;
  
  
}


void test_4() {
  // my personal settings:
  // export VALKKA_TEST_SDP=/home/sampsa/C/valkka/aux/multicast.sdp
  const char* name = "@TEST: opengl_thread_test: test 4: ";
  std::cout << name <<"** @@Test basic chain **" << std::endl;
  
  if (!stream_sdp) {
    std::cout << name <<"ERROR: missing test sdp stream: set environment variable VALKKA_TEST_SDP"<< std::endl;
    exit(2);
  }
  std::cout << name <<"** test sdp stream: "<< stream_sdp << std::endl;
  
  int i;
  // ***********************************
  // filtergraph:
  // (LiveThread:livethread) --> BasicChain --> {FifoFrameFilter:gl_in_gilter} --> [OpenGLFrameFifo:gl_fifo] -->> (OpenGLThread:glthread)
  //
  OpenGLThread      glthread        ("glthread",/*n720p*/10,/*n1080p*/10,/*n1440p*/0,/*4K*/0,/*naudio*/10,/*msbuftime*/100,/*core_id*/-1); // remember buffering time!
  OpenGLFrameFifo&  gl_fifo         =glthread.getFifo();      // get gl_fifo from glthread
  FifoFrameFilter   gl_in_filter    ("gl_in_filter",gl_fifo);   
  
  BasicChain chain(LiveConnectionType::sdp, stream_sdp, 1, gl_in_filter); // BasicChain(LiveConnectionType contype, const char* adr, SlotNumber n_slot, FifoFrameFilter& gl_in_filter);
  
  LiveThread        livethread      ("livethread");
  // ***********************************
  
  std::cout << name << "starting threads" << std::endl;
  glthread.startCall(); // start running OpenGLThread!
  
  Window window_id  =glthread.createWindow();
  glthread.makeCurrent(window_id);
  std::cout << "new x window "<<window_id<<std::endl;
  
  livethread.      startCall();
  chain.avthread-> startCall();
  chain.avthread-> decodingOnCall();
  
  std::cout << name << "registering stream" << std::endl;
  livethread.registerStreamCall(chain.getCtx());
  livethread.playStreamCall(chain.getCtx());
  
  glthread.newRenderGroupCall(window_id); 
  sleep_for(1s);
  i=glthread.newRenderContextCall(1, window_id, 0); std::cout << "got render context id "<<i<<std::endl; 
  
  std::cout << name << "wait" << std::endl;
  sleep_for(5s);
  std::cout << name << "waited" << std::endl;
  
  glthread.delRenderContextCall(i);
  glthread.delRenderGroupCall(window_id);
  
  std::cout << name << "stopping threads" << std::endl;
  chain.avthread->stopCall();
  livethread. stopCall();
  glthread.   stopCall();
  std::cout << name << "All threads stopped" << std::endl;
  sleep_for(1s);
  std::cout << name << "Leaving context" << std::endl;
}


void test_5() {
  int n_streams=2;
  int n_stack  =20;
  
  const char* name = "@TEST: opengl_thread_test: test 5: ";
  std::cout << name <<"** @@OpenGL live rendering with N sdp streams.  Just for 5 sec **" << std::endl;
  
  if (!stream_sdp) {
    std::cout << name <<"ERROR: missing test sdp stream: set environment variable VALKKA_TEST_SDP"<< std::endl;
    exit(2);
  }
  std::cout << name <<"** test sdp stream: "<< stream_sdp << std::endl;
  
  int i;
  // filtergraph:
  // (LiveThread:livethread) --> BasicChain --> {FifoFrameFilter:gl_in_gilter} --> [OpenGLFrameFifo:gl_fifo] -->> (OpenGLThread:glthread)
  //
  OpenGLThread      glthread        ("glthread",/*n720p*/n_stack,/*n1080p*/n_stack,/*n1440p*/0,/*4K*/0,/*naudio*/10,/*msbuftime*/100,/*core_id*/-1); // remember buffering time!
  OpenGLFrameFifo&  gl_fifo         =glthread.getFifo();      // get gl_fifo from glthread
  FifoFrameFilter   gl_in_filter    ("gl_in_filter",gl_fifo);   
  // BasicChain chain(LiveConnectionType::sdp, stream_sdp, 1, gl_in_filter); // BasicChain(LiveConnectionType contype, const char* adr, SlotNumber n_slot, FifoFrameFilter& gl_in_filter);
  LiveThread        livethread      ("livethread");
  
  std::vector<BasicChain*> chains;
  
  /*
  std::vector<BasicChain> chains;
  int c;
  for(c=0;c<n_streams;c++) {
    chains.push_back(BasicChain(LiveConnectionType::sdp, stream_sdp, c, gl_in_filter));
    // BasicChain instantiated => call new on AVThread => valid AVThread with has_thread=true .. BasicChain copy-cnstr called .. takes a copy of AVThread pointer ..
    // .. std::vector does the following: it copy-constructs a new BasicChain into the vector.  Once we go out of scope from here (at "}"), that "auxiliary" BasicChain 
    // we have written in "chains.push_back(BasicChain(..))" get's deleted .. rendering the pointers invalid.
  }
  // in any case, we wan't to do these things in python (that reference counts the pointers automatically) not in cpp
  */
  
  int c;
  for(c=0;c<n_streams;c++) {
    chains.push_back(new BasicChain(LiveConnectionType::sdp, stream_sdp, c, gl_in_filter));
  }
  
  std::cout << name << "starting threads" << std::endl;
  glthread.  startCall(); // start running OpenGLThread!
  livethread.startCall();
  
  for(auto it=chains.begin(); it!=chains.end(); ++it) {
    Window window_id  =glthread.createWindow();
    glthread.makeCurrent(window_id);
    std::cout << "new x window "<<window_id<<std::endl;
    (*it)->avthread->startCall();
    (*it)->avthread->decodingOnCall();
    (*it)->setWindow(window_id);
  }
  
  std::cout << name << "registering streams" << std::endl;
  
  for(auto it=chains.begin(); it!=chains.end(); ++it) {
    livethread.registerStreamCall((*it)->getCtx());
    livethread.playStreamCall((*it)->getCtx());
  }
  
  for(auto it=chains.begin(); it!=chains.end(); ++it) {
    glthread.newRenderGroupCall((*it)->getWindow()); 
    i=glthread.newRenderContextCall((*it)->getSlot(), (*it)->getWindow(), 0); std::cout << "got render context id "<<i<<std::endl;
    (*it)->setRenderCtx(i);
  }
  
  sleep_for(5s);
  
  for(auto it=chains.begin(); it!=chains.end(); ++it) {
    glthread.delRenderContextCall((*it)->getRenderCtx());
    glthread.delRenderGroupCall((*it)->getWindow());
  }
  
  for(auto it=chains.begin(); it!=chains.end(); ++it) {
    (*it)->avthread->stopCall();
  }
  
  std::cout << name << "stopping threads" << std::endl;
  livethread.      stopCall();
  glthread.        stopCall();
  std::cout << name << "All threads stopped" << std::endl;
  std::cout << name << "Releasing.." << std::endl;
  sleep_for(1s);
  for(c=0;c<n_streams;c++) {
    delete chains[c];
  }
  std::cout << name << "Leaving context" << std::endl;  
}


void test_6() {
  int n_streams=3;
  int n_stack  =n_streams*10;
  
  const char* name = "@TEST: opengl_thread_test: test 6: ";
  std::cout << name <<"** @@OpenGL live rendering with N sdp streams.  For N secs **" << std::endl;
  
  if (!stream_sdp) {
    std::cout << name <<"ERROR: missing test sdp stream: set environment variable VALKKA_TEST_SDP"<< std::endl;
    exit(2);
  }
  std::cout << name <<"** test sdp stream: "<< stream_sdp << std::endl;
  
  int i;
  // filtergraph:
  // (LiveThread:livethread) --> BasicChain --> {FifoFrameFilter:gl_in_gilter} --> [OpenGLFrameFifo:gl_fifo] -->> (OpenGLThread:glthread)
  //
  OpenGLThread      glthread        ("glthread",/*n720p*/n_stack,/*n1080p*/n_stack,/*n1440p*/0,/*4K*/0,/*naudio*/10,/*msbuftime*/100,/*core_id*/1); // remember buffering time!
  OpenGLFrameFifo&  gl_fifo         =glthread.getFifo();      // get gl_fifo from glthread
  FifoFrameFilter   gl_in_filter    ("gl_in_filter",gl_fifo);   
  // BasicChain chain(LiveConnectionType::sdp, stream_sdp, 1, gl_in_filter); // BasicChain(LiveConnectionType contype, const char* adr, SlotNumber n_slot, FifoFrameFilter& gl_in_filter);
  LiveThread        livethread      ("livethread");
  
  std::vector<BasicChain*> chains;
  
  /*
  std::vector<BasicChain> chains;
  int c;
  for(c=0;c<n_streams;c++) {
    chains.push_back(BasicChain(LiveConnectionType::sdp, stream_sdp, c, gl_in_filter));
    // BasicChain instantiated => call new on AVThread => valid AVThread with has_thread=true .. BasicChain copy-cnstr called .. takes a copy of AVThread pointer ..
    // .. std::vector does the following: it copy-constructs a new BasicChain into the vector.  Once we go out of scope from here (at "}"), that "auxiliary" BasicChain 
    // we have written in "chains.push_back(BasicChain(..))" get's deleted .. rendering the pointers invalid.
  }
  // in any case, we wan't to do these things in python (that reference counts the pointers automatically) not in cpp
  */
  
  int c;
  for(c=0;c<n_streams;c++) {
    chains.push_back(new BasicChain(LiveConnectionType::sdp, stream_sdp, c, gl_in_filter));
  }
  
  std::cout << name << "starting threads" << std::endl;
  glthread.  startCall(); // start running OpenGLThread!
  livethread.startCall();
  
  for(auto it=chains.begin(); it!=chains.end(); ++it) {
    Window window_id  =glthread.createWindow();
    glthread.makeCurrent(window_id);
    std::cout << "new x window "<<window_id<<std::endl;
    (*it)->avthread->startCall();
    (*it)->avthread->decodingOnCall();
    (*it)->setWindow(window_id);
  }
  
  std::cout << name << "registering streams" << std::endl;
  
  for(auto it=chains.begin(); it!=chains.end(); ++it) {
    livethread.registerStreamCall((*it)->getCtx());
    livethread.playStreamCall((*it)->getCtx());
  }
  
  for(auto it=chains.begin(); it!=chains.end(); ++it) {
    glthread.newRenderGroupCall((*it)->getWindow()); 
    i=glthread.newRenderContextCall((*it)->getSlot(), (*it)->getWindow(), 0); std::cout << "got render context id "<<i<<std::endl;
    (*it)->setRenderCtx(i);
  }
  
  sleep_for(10s);
  
  for(auto it=chains.begin(); it!=chains.end(); ++it) {
    glthread.delRenderContextCall((*it)->getRenderCtx());
    glthread.delRenderGroupCall((*it)->getWindow());
    (*it)->avthread->stopCall();
  }
  
  std::cout << name << "stopping threads" << std::endl;
  livethread.      stopCall();
  glthread.        stopCall();
  std::cout << name << "All threads stopped" << std::endl;
  std::cout << name << "Releasing.." << std::endl;
  sleep_for(1s);
  for(c=0;c<n_streams;c++) {
    delete chains[c];
  }
  std::cout << name << "Leaving context" << std::endl;  
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
      case(6):
        test_6();
        break;
      default:
        std::cout << "No such test "<<argcv[1]<<" for "<<argcv[0]<<std::endl;
    }
  }
} 




