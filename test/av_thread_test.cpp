/*
 * av_thread_test.cpp : Test class AVThread
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
 *  @file    av_thread_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
 *  
 *  @brief Test class AVThread
 *
 */ 

#include "avthread.h"
#include "filters.h"
#include "logging.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for;  // http://en.cppreference.com/w/cpp/thread/sleep_for

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
  const char* name = "@TEST: av_thread_test: test 1: ";
  std::cout << name <<"** @@Send a setup frame to AVThread **" << std::endl;
  
  Frame setupframe =Frame();
  Frame vframe     =Frame();
  
  // *****
  // filtergraph:
  // [FrameFifo:in_fifo] -->> [AVThread:avthread] --> {FrameFilter:out_filter}
  FrameFifo in_fifo ("in_fifo",10);
  InfoFrameFilter out_filter("out");
  AVThread avthread("avthread",in_fifo,out_filter); // &out_fifo);
  // *****
  
  vframe.frametype =FrameType::h264;
  vframe.subsession_index =0;
  vframe.payload.resize(10);
  vframe.payload={0,0,0,1,0,0,0,0,0,0};
  
  setupframe.frametype =FrameType::setup; // this is a setup frame
  setupframe.setup_pars.frametype =FrameType::h264; // what frame types are to be expected from this stream
  setupframe.subsession_index =0;
  
  std::cout << name << "starting av thread" << std::endl;
  avthread.startCall();
  avthread.decodingOnCall();
  
  in_fifo.dumpStack();
  
  sleep_for(2s);
  
  in_fifo.writeCopy(&setupframe);
  
  sleep_for(5s);
  
  in_fifo.dumpStack();
  
  std::cout << name << "stopping live thread" << std::endl;
  avthread.stopCall();
}


void test_2() {
  const char* name = "@TEST: av_thread_test: test 2: ";
  std::cout << name <<"** @@Send a setup frame and two void video frames to AVThread **" << std::endl;
  
  Frame setupframe =Frame();
  Frame vframe     =Frame();
  
  // *****
  // filtergraph:
  // [FrameFifo:in_fifo] -->> [AVThread:avthread] --> {FrameFilter:out_filter}
  FrameFifo in_fifo ("in_fifo",10);
  InfoFrameFilter out_filter("out");
  AVThread avthread("avthread",in_fifo,out_filter); // &out_fifo);
  // ******
  
  vframe.frametype =FrameType::h264;
  vframe.subsession_index =0;
  // vframe.reserve(10);
  vframe.payload.resize(10);
  vframe.payload={0,0,0,1,0,0,0,0,0,0};
  
  setupframe.frametype =FrameType::setup; // this is a setup frame
  setupframe.setup_pars.frametype =FrameType::h264; // what frame types are to be expected from this stream
  setupframe.subsession_index =0;
  
  std::cout << name << "starting av thread" << std::endl;
  avthread.startCall();
  avthread.decodingOnCall();
  
  in_fifo.dumpStack();
  
  sleep_for(2s);
  
  std::cout << name << "\nWriting setup frame\n";
  in_fifo.writeCopy(&setupframe);
  std::cout << name << "\nWriting H264 frame\n";
  in_fifo.writeCopy(&vframe);
  std::cout << name << "\nWriting H264 frame\n";
  in_fifo.writeCopy(&vframe);
  
  sleep_for(5s);
  
  in_fifo.dumpStack();
  
  std::cout << name << "stopping live thread" << std::endl;
  avthread.stopCall();
}


void test_3() {
  const char* name = "@TEST: av_thread_test: test 3: ";
  std::cout << name <<"** @@Send a void video frame to AVThread (no setup frame) **" << std::endl;
  
  Frame setupframe =Frame();
  Frame vframe     =Frame();
  
  // *****
  // filtergraph:
  // [FrameFifo:in_fifo] -->> [AVThread:avthread] --> {FrameFilter:out_filter}
  FrameFifo in_fifo ("in_fifo",10);
  InfoFrameFilter out_filter("out");
  AVThread avthread("avthread",in_fifo,out_filter); // &out_fifo);
  // *****
  
  vframe.frametype =FrameType::h264;
  vframe.subsession_index =0;
  // vframe.reserve(10);
  vframe.payload.resize(10);
  vframe.payload={0,0,0,1,0,0,0,0,0,0};
  
  setupframe.frametype =FrameType::setup; // this is a setup frame
  setupframe.setup_pars.frametype =FrameType::h264; // what frame types are to be expected from this stream
  setupframe.subsession_index =0;
  
  std::cout << name << "starting av thread" << std::endl;
  avthread.startCall();
  avthread.decodingOnCall();
  
  in_fifo.dumpStack();
  
  sleep_for(2s);
  
  in_fifo.writeCopy(&vframe);
  in_fifo.writeCopy(&vframe);
  
  sleep_for(5s);
  
  in_fifo.dumpStack();
  
  std::cout << name << "stopping live thread" << std::endl;
  avthread.stopCall();
}


void test_4() {
  const char* name = "@TEST: av_thread_test: test 4: ";
  std::cout << name <<"** @@Send two consecutive setup frames to AVThread (i.e. reinit) **" << std::endl;
  
  Frame setupframe =Frame();
  Frame vframe     =Frame();
  
  // *****
  // filtergraph:
  // [FrameFifo:in_fifo] -->> [AVThread:avthread] --> {FrameFilter:out_filter}
  FrameFifo in_fifo ("in_fifo",10);
  InfoFrameFilter out_filter("out");
  AVThread avthread("avthread",in_fifo,out_filter);
  // *****
  
  vframe.frametype =FrameType::h264;
  vframe.subsession_index =0;
  vframe.payload.resize(10);
  vframe.payload={0,0,0,1,0,0,0,0,0,0};
  
  setupframe.frametype =FrameType::setup; // this is a setup frame
  setupframe.setup_pars.frametype =FrameType::h264; // what frame types are to be expected from this stream
  setupframe.subsession_index =0;
  
  std::cout << name << "starting av thread" << std::endl;
  avthread.startCall();
  avthread.decodingOnCall();
  
  in_fifo.dumpStack();
  
  sleep_for(2s);
  
  in_fifo.writeCopy(&setupframe);
  in_fifo.writeCopy(&setupframe);
  in_fifo.writeCopy(&vframe);
  in_fifo.writeCopy(&vframe);
  
  sleep_for(5s);
  
  in_fifo.dumpStack();
  
  std::cout << name << "stopping live thread" << std::endl;
  avthread.stopCall();
}


void test_5() {
  const char* name = "@TEST: av_thread_test: test 5: ";
  std::cout << name <<"** @@Start two AVThreads **" << std::endl;
  
  Frame setupframe =Frame();
  Frame vframe     =Frame();
  
  // *****
  // filtergraph:
  // [FrameFifo:infifo]  -->> [AVThread:avthread]  --> {FrameFilter:out_filter}
  // [FrameFifo:infifo2] -->> [AVThread:avthread2] --> {FrameFilter:out_filter2}
    
  FrameFifo in_fifo  ("in_fifo",10);
  InfoFrameFilter out_filter("out");
  
  FrameFifo in_fifo2 ("in_fifo2",10);
  InfoFrameFilter out_filter2("out");  
  
  AVThread avthread ("[avthread 1]",in_fifo, out_filter);  // &out_fifo);
  AVThread avthread2("[avthread 2]",in_fifo2,out_filter2); //&out_fifo2);
  // *****
  
  vframe.frametype =FrameType::h264;
  vframe.subsession_index =0;
  vframe.payload.resize(10);
  vframe.payload={0,0,0,1,0,0,0,0,0,0};
  
  setupframe.frametype =FrameType::setup; // this is a setup frame
  setupframe.setup_pars.frametype =FrameType::h264; // what frame types are to be expected from this stream
  setupframe.subsession_index =0;
  
  std::cout << name << "starting av thread" << std::endl;
  
  avthread. startCall();
  avthread2.startCall();
  
  avthread. decodingOnCall();
  avthread2.decodingOnCall();
  
  in_fifo.dumpStack();
  
  sleep_for(2s);
  
  in_fifo.writeCopy(&setupframe);
  
  sleep_for(5s);
  
  in_fifo.dumpStack();
  
  std::cout << name << "stopping live thread" << std::endl;
  
  avthread. stopCall();
  avthread2.stopCall();
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

