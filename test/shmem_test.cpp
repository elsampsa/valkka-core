/*
 * shmem_test.cpp : Test shared memory classes for multiprocessing
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
 *  @file    shmem_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.11.0 
 *  
 *  @brief 
 *
 */ 

#include "avdep.h"
#include "logging.h"
#include "livethread.h"
#include "sharedmem.h"


using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() { // open two terminals, start this from terminal 1 and "test 2" from terminal 2
  int inp;
  const char* name = "@TEST: shmem_test: test 1: ";
  std::cout << name <<"** @@Create shared memory on the server side : INTERACTIVE TEST **" << std::endl;
  
  std::vector<uint8_t> payload;
  SharedMemSegment shmem("testing", 30*1024*1024, true);
  
  payload.resize(10,0);
  
  std::cout << "Enter an integer to write into shared mem" << std::endl;
  std::cin >> inp;
  for (auto it=payload.begin(); it!=payload.end(); ++it) {
    *it=(uint8_t)inp;
  }
  shmem.put(payload);
  
  std::cout << "Enter an integer. 0 to exit" << std::endl;
  std::cin >> inp;
}


void test_2() {
  int inp, i, n;
  const char* name = "@TEST: shmem_test: test 2: ";
  std::cout << name <<"** @@Create shared memory on the client side : INTERACTIVE TEST **" << std::endl;
  
  SharedMemSegment shmem("testing", 30*1024*1024, false);
  std::cout << "Enter an integer to read from shared mem" << std::endl;
  std::cin >> inp;
  n=shmem.getSize();
  std::cout << "Amount of bytes in shmem buffer: "<<n<<std::endl;
  for(i=0; i<n; i++) {
   std::cout << int(shmem.payload[i]) << std::endl;
  }
  
}


void test_3() { // open two terminals.  Start this from first terminal and test_4 from the second terminal
  const char* name = "@TEST: shmem_test: test 3: ";
  std::cout << name <<"** @@Create shmem ring buffer on the server side : INTERACTIVE TEST **" << std::endl;
  int inp, cc, i, index;
  std::vector<uint8_t> payload;
  SharedMemRingBuffer rb("testing",10,30*1024*1024,1000,true); // name, ncells, bytes per cell, timeout, server or not
  
  cc=0;
  payload.resize(10,cc);
  std::cout << "Give number of cells to write. 0 = exit"<<std::endl;
  while (true) {
    std::cout << std::endl << "> ";
    std::cin >> inp;
    if (inp==0) { 
      break; 
    }
    
    for(i=0;i<inp;++i) {
      
      for (auto it=payload.begin(); it!=payload.end(); ++it) {
        *it=(uint8_t)cc;
      }
      
      /* // not here.. index still -1
      std::cout << "Wrote "<<inp<<" cells with byte values="<<cc<<std::endl;
      index=rb.getValue()-1;
      std::cout << "Data at index " << index << " : " << int(rb.shmems[index]->payload[0]) << " " << int(rb.shmems[index]->payload[1]) << " " << int(rb.shmems[index]->payload[2]) << std::endl;
      */
      
      rb.serverPush(payload);
      /* // if we peek the shmem's now, there seems to be some sort of race condition, as the client will access it at exactly the same time.. => segfault
      std::cout << "Wrote "<<inp<<" cells with byte values="<<cc<<std::endl;
      index=rb.getValue()-1;
      std::cout << "Data at index " << index << " : " << int(rb.shmems[index]->payload[0]) << " " << int(rb.shmems[index]->payload[1]) << " " << int(rb.shmems[index]->payload[2]) << std::endl;
      */
      
      cc++;
      std::cout << std::endl;
    }
  }
}


void test_4() {
  const char* name = "@TEST: shmem_test: test 4: ";
  std::cout << name <<"** @@Create shmem ring buffer on the client side : INTERACTIVE TEST **" << std::endl;
  int inp, index, i, ii, n;
  bool ok;
  std::vector<uint8_t> payload;
  SharedMemRingBuffer rb("testing",10,30*1024*1024,5000,false); // name, ncells, bytes per cell, timeout, server or not
  
  payload.resize(10,0);
  std::cout << "Give number of cells to read. 0 = exit"<<std::endl;
  while (true) {
    std::cout << std::endl << "> ";
    std::cin >> inp;
    if (inp==0) { 
      break; 
    }
    for(i=0;i<inp;++i) {
      ok=rb.clientPull(index, n);
      if (ok) {
        // n=rb.shmems[index]->getSize();
        std::cout << "cell index " << index << " has " << n << " bytes" << std::endl;
        for(ii=0; ii<n; ii++) {
          std::cout << int(rb.shmems[index]->payload[ii]) << std::endl;
        }
        std::cout << std::endl;
        }
      else {
        std::cout << "semaphore timed out!" << std::endl;
        std::cout << std::endl;
      }
    }
  }
}


void test_5() {
  // TODO: make a hard-core shmem ringbuffer write/read test - random reads/writes within milliseconds
  const char* name = "@TEST: shmem_test: test 5: ";
  std::cout << name <<"** @@DESCRIPTION : TODO **" << std::endl;
}


void test_6() { // open two terminals.  Start this from first terminal and test_4 from the second terminal
  const char* name = "@TEST: shmem_test: test 6: ";
  std::cout << name <<"** @@Test ShmemFrameFilter ** : INTERACTIVE TEST " << std::endl;
  
  BasicFrame *f = new BasicFrame;
  ShmemFrameFilter shmem("testing", 10, 1024*1024*30);
  // SharedMemFrameFilter(const char* name, int n_cells, std::size_t n_bytes, int mstimeout)
  
  int inp, cc, i, index;
  std::vector<uint8_t> payload;
  // SharedMemRingBuffer rb("testing",10,30*1024*1024,1000,true); // name, ncells, bytes per cell, timeout, server or not
  
  cc=0;
  payload.resize(10,cc);
  std::cout << "Give number of cells to write. 0 = exit"<<std::endl;
  while (true) {
    std::cout << std::endl << "> ";
    std::cin >> inp;
    if (inp==0) { 
      break; 
    }
    
    for(i=0;i<inp;++i) {
      
      for (auto it=payload.begin(); it!=payload.end(); ++it) {
        *it=(uint8_t)cc;
      }
      
      //rb.serverPush(payload);
      f->payload=payload; // copies bytes to frame's payload
      shmem.run(f);
      
      cc++;
      std::cout << std::endl;
    }
  }
  
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


