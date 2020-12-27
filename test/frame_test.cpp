/*
 * frames_test.cpp : Testing the Frame classes.
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
 *  @file    frame_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.0.3 
 *  
 *  @brief Testing Frame classes.  Compile with "make tests" and run with valgrind
 *
 */

#include "frame.h"
#include "framefilter.h"
#include "logging.h"
#include "avdep.h"
#include "test_import.h"


const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
  const char* name = "@TEST: frames_test: test 1: ";
  std::cout << name <<"** @@Create frame **" << std::endl;
  
  BasicFrame* f;

  f=new BasicFrame();
  f->reserve(1024*1024*5);
  delete f;

}


void test_2() {
  const char* name = "@TEST: frames_test: test 2: ";
  std::cout << name <<"** @@Frame filters **" << std::endl;

  FrameFilter* f1;
  FrameFilter* f2;
  FrameFilter* f3;
  BasicFrame* f;

  f3 = new DummyFrameFilter("filter 3");
  f2 = new DummyFrameFilter("filter 2", f3);
  f1 = new DummyFrameFilter("filter 1", f2);

  f=new BasicFrame();
  f->reserve(1024*1024*5);

  f1->run(f);
  
  delete f;
  delete f1;
  delete f2;
  delete f3;
}


void test_3() {
    const char* name = "@TEST: frames_test: test 3: ";
    std::cout << name <<"** @@Copy RGB frames **" << std::endl;

    AVRGBFrame *f1 = new AVRGBFrame();
    AVRGBFrame *f2 = new AVRGBFrame();
    
    f1->reserve(720,1920);
    f2->reserve(720,1920);
    
    std::cout << "f1: " << *f1 << std::endl;
    std::cout << "f2: " << *f2 << std::endl;
    
    f1->copyPayloadFrom(f2);
    
    delete f1;
    delete f2;
}


void test_4() {
    const char* name = "@TEST: frames_test: test 4: ";
    std::cout << name <<"** @@Test copying signalframes **" << std::endl;

    SignalFrame *f1 = new SignalFrame();
    SignalFrame *f2 = new SignalFrame();

    OfflineSignalContext signal_ctx = OfflineSignalContext();
    OfflineSignalContext signal_ctx2 = OfflineSignalContext();

    signal_ctx.n_slot = 10;
    std::cout << "n_slot in " << signal_ctx.n_slot << std::endl;

    put_signal_context(f1, signal_ctx);
    f2->copyFrom(f1);
    get_signal_context(f2, signal_ctx2);

    std::cout << "n_slot out " << signal_ctx2.n_slot << std::endl;

    delete f1;
    delete f2;
}


void test_5() {
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


