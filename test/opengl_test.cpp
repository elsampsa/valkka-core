/*
 * opengl_test.cpp : Testing OpenGL calls, without threading
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
 *  @file    opengl_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.5 
 *  
 *  @brief Testing OpenGL calls, without threading
 *
 */ 

#include "openglthread.h"
#include "filters.h"
#include "logging.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
  const char* name = "@TEST: opengl_test: test 1: ";
  std::cout << name <<"** @@Compile shaders **" << std::endl;

  OpenGLThread glthread("glthread",/*n720p*/10,/*n1080p*/0,/*n1440p*/0,/*4K*/0,/*msbuftime*/0,/*core_id*/-1);
  // const char* name, unsigned short n720p=0, unsigned short n1080p=0, unsigned short n1440p=0, unsigned short n4K=0, unsigned msbuftime=100, int core_id=-1
  glthread.preRun();
  glthread.postRun();
}


void test_2() {
  const char* name = "@TEST: opengl_test: test 2: ";
  std::cout << name <<"** @@Open a window **" << std::endl;
  
  Window window_id;
  OpenGLThread glthread("glthread",/*n720p*/10,/*n1080p*/0,/*n1440p*/0,/*4K*/0,/*msbuftime*/0,/*core_id*/-1);
  
  glthread.preRun();
  
  window_id=glthread.createWindow();
  glthread.makeCurrent(window_id);
  
  sleep_for(10s);
  
  glthread.postRun();
  
}


void test_3() {
  const char* name = "@TEST: opengl_test: test 3: ";
  std::cout << name <<"** @@Reserve PBOs **" << std::endl;
  
  OpenGLThread glthread("glthread",/*n720p*/10,/*n1080p*/0,/*n1440p*/10,/*4K*/0,/*msbuftime*/0,/*core_id*/-1);

  glthread.preRun();
  
  glthread.reportStacks();
  
  glthread.postRun();
}


void test_4() {
  const char* name = "@TEST: opengl_test: test 4: ";
  
  OpenGLThread glthread("glthread",/*n720p*/10,/*n1080p*/0,/*n1440p*/10,/*4K*/0,/*msbuftime*/0,/*core_id*/-1);
  
  glthread.preRun();
  
  glthread.reportStacks();
  
  glthread.postRun();
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




