/*
 * ringbuffer_test.cpp :
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
 *  @file    ringbuffer_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.5.1 
 *  
 *  @brief 
 *
 */ 

#include "framefifo.h"
#include "framefilter.h"
#include "logging.h"
#include "sharedmem.h"
#include "test_import.h"


using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
  const char* name = "@TEST: ringbuffer_test: test 1: ";
  std::cout << name <<"** @@Standalone (server & client) sharedmem test for simple shmem frames ** **" << std::endl;
  
  int index_out = 0;
  int meta;
  bool ok;
  BasicFrame *f = new BasicFrame();
  f->resize(300);
  
  std::cout << "reserved frame " << *f << std::endl;
  
  // const char* name, int n_cells, int width, int height, int mstimeout=0, bool is_server=false
  SharedMemRingBuffer server = SharedMemRingBuffer("kokkelis", 10, 300, 0, true);
  SharedMemRingBuffer client = SharedMemRingBuffer("kokkelis", 10, 300, 0, false);
  
  server.serverPush(f->payload);
  server.serverPush(f->payload);
  server.serverPush(f->payload);
  
  ok = client.clientPull(index_out, meta);
  std::cout << "index_out = " << index_out << std::endl;  
  std::cout << "size = " << meta << std::endl;
  
  ok = client.clientPull(index_out, meta);
  std::cout << "index_out = " << index_out << std::endl;  
  std::cout << "size = " << meta << std::endl;
  
  ok = client.clientPull(index_out, meta);
  std::cout << "index_out = " << index_out << std::endl;  
  std::cout << "size = " << meta << std::endl;
  
  delete f;
}


void test_2() {
  const char* name = "@TEST: ringbuffer_test: test 2: ";
  std::cout << name <<"** @@Standalone (server & client) sharedmem test for RGB24 frames **" << std::endl;
  
  int index_out = 0;
  RGB24Meta meta = RGB24Meta();
  bool ok;
  AVRGBFrame *f = new AVRGBFrame();
  f->reserve(300, 300);
  
  std::cout << "reserved frame " << *f << std::endl;
  
  // const char* name, int n_cells, int width, int height, int mstimeout=0, bool is_server=false
  SharedMemRingBufferRGB server = SharedMemRingBufferRGB("kokkelis", 10, 300, 300, 0, true);
  SharedMemRingBufferRGB client = SharedMemRingBufferRGB("kokkelis", 10, 300, 300, 0, false);
  
  f->mstimestamp = 1000;
  f->n_slot = 1;
  server.serverPushAVRGBFrame(f);
  
  f->mstimestamp = 2000;
  f->n_slot = 2;
  server.serverPushAVRGBFrame(f);
  
  f->mstimestamp = 3000;
  f->n_slot = 3;
  server.serverPushAVRGBFrame(f);
  
  ok = client.clientPullFrame(index_out, meta);
  std::cout << "index_out = " << index_out << std::endl;  
  std::cout << "meta: size:" << meta.size << " width:" << meta.width << " height:" << meta.height << " slot:" << meta.slot << std::endl;
  
  ok = client.clientPullFrame(index_out, meta);
  std::cout << "index_out = " << index_out << std::endl;  
  std::cout << "meta: size:" << meta.size << " width:" << meta.width << " height:" << meta.height << " slot:" << meta.slot << std::endl;
  
  /*
  ok = client.clientPull2(index_out, meta);
  std::cout << "index_out = " << index_out << std::endl;  
  std::cout << "meta: size:" << meta.size << " width:" << meta.width << " height:" << meta.height << " slot:" << meta.slot << std::endl;
  */
  // test the old API as well:
  int size;
  
  ok = client.clientPull(index_out, size);
  std::cout << "index_out = " << index_out << std::endl;  
  std::cout << "size = " << size << std::endl;
  
  delete f;
}


void test_3() {
  
  const char* name = "@TEST: ringbuffer_test: test 3: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}


void test_4() {
  
  const char* name = "@TEST: ringbuffer_test: test 4: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}


void test_5() {
  
  const char* name = "@TEST: ringbuffer_test: test 5: ";
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


