/*
 * fifo_test.cpp : Testing the fifo classes
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
 *  @file    fifo_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.6 
 *  
 *  @brief Testing fifo classes.  Compile with "make tests" and run with valgrind
 *
 */

#include "queues.h"
#include "filters.h"
#include "logging.h"

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
  const char* name = "@TEST: fifo_test: test 1: ";
  std::cout << name <<"** @@Basic fifo tests **" << std::endl;
  
  FrameFifo fifo("fifo",10);
  Frame* f;
  Frame f2;
  Frame* newframe;

  fifo.dumpStack();
  std::cout << std::endl;

  ///*
  f=fifo.stack.front();
  std::cout <<f << std::endl; // adr
  std::cout <<f->getMsTimestamp() << std::endl; // timestamp
  std::cout <<f->payload.size() << std::endl; // 0

  f=fifo.stack[3];
  std::cout <<f << std::endl; // adr
  std::cout <<f->getMsTimestamp() << std::endl; // timestamp
  std::cout <<f->payload.size() << std::endl; // 0
  //*/
    
  f2=*(f); // a deep copy (replicates byte data)

  std::cout << f2;

  newframe = new Frame();

  fifo.writeCopy(newframe);

  fifo.dumpStack();
  std::cout << std::endl;
  fifo.dumpFifo();
  std::cout << std::endl;

  delete newframe;
  
}


void test_2() {
  const char* name = "@TEST: fifo_test: test 2: ";
  std::cout << name <<"** @@Fifo overflow **" << std::endl;
  
  FrameFifo fifo("fifo",10);
  Frame* f;
  uint i;
  bool res;
  f = new Frame();

  for(i=0; i<20; i++) {
    res=fifo.writeCopy(f);
    std::cout << "writeCopy " << i << " returned " << res << std::endl;
    fifo.dumpFifo();
    }

  delete f;
  
}


void test_3() {
}


void test_4() {
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


