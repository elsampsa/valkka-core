/*
 * cachestream_decode_test.cpp : From cachestream to decoder
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
 *  @file    cachestream_decode_test.cpp
 *  @author  Petri Eranko
 *  @date    2018
 *  @version 1.3.3 
 *  
 *  @brief 
 *
 */ 

#include "framefifo.h"
#include "framefilter.h"
#include "logging.h"
#include "avdep.h"
#include "valkkafs.h"
#include "valkkafsreader.h"
#include "cachestream.h"
#include "avthread.h"
#include "test_import.h"


using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
    const char* name = "@TEST: cachestream_decode_test: test 1: ";
    std::cout << name <<"** @@Use saved filesystem with id 1.  Test seek.  Pass to AVThread**" << std::endl;
    // ValkkaFSReaderThread (freader) => FileCacheThread (fcacher) => in_filter : AVThread (avhread)
    
    // InfoFrameFilter decoded_info("decoded", &gl_in_filter);
    DummyFrameFilter decoded_info("decoded", true);
    AVThread         avthread("avthread", decoded_info);
    
    FifoFrameFilter  &in_filter = avthread.getFrameFilter(); // request framefilter from AVThread
    // InfoFrameFilter in_filter("encoded");
    
    int n_blocks = 10;
    int i;
    
    // create ValkkaFS and WriterThread.  15 frames per block, 50 blocks
    ValkkaFS fs("/dev/sda1", "/home/sampsa/python3_packages/valkka_examples/api_level_2/qt/fs_directory/blockfile", 2097152, n_blocks, false); // dumpfile, blockfile, blocksize, number of blocks, clear blocktable
    FileCacheThread fcacher("cacher");
    ValkkaFSReaderThread freader("reader", fs, fcacher.getFrameFilter()); // freader => fcacher
    
    avthread.startCall();
    fcacher.startCall();
    freader.startCall();
    
    avthread.decodingOnCall();
    
    freader.setSlotIdCall(1, 1); // map slot 1 from id 1
    FileStreamContext ctx(1, &in_filter); // slot, framefilter
    fcacher.registerStreamCall(ctx); // slot 1 to in_filter
    
    ValkkaFSTool fstool(fs);
    
    for(i=0;i<=n_blocks;i++) {
        fstool.dumpBlock(i);
    }
    
    long int mstime;
    
    // TEST 1
    /*
    std::list<std::size_t> block_list = {0, 1, 2, 3, 4, 5};
    freader.pullBlocksCall(block_list); // send frames.  Once the transmission ends, seek iteration is performed
    
    // now the new frames are available
    sleep_for(2s);
    
    // WARNING: remember to seek to a time that's withing your requested blocks!
    mstime = 1554397786825; // just after key-frames @ block 3
    
    fcacher.seekStreamsCall(mstime); // set the target time
    sleep_for(4s);
    
    mstime = mstime + 3000;
    
    fcacher.seekStreamsCall(mstime); // set the target time
    sleep_for(4s);
    */
    
    // TEST 2
    mstime = 1554397757968;
    fcacher.seekStreamsCall(mstime);
    std::list<std::size_t> block_list = {0, 1, 9};
    freader.pullBlocksCall(block_list); // send frames.  Once the transmission ends, seek iteration is performed

    sleep_for(5s);
    
    freader.stopCall();
    fcacher.stopCall();
    avthread.stopCall();
}



void test_2() {
    const char* name = "@TEST: cachestream_decode_test: test 2: ";
    std::cout << name <<"** @@Use saved filesystem, with id's 1, 2, 3.  Test seek.  Pass to AVThread **" << std::endl;
    // ValkkaFSReaderThread (freader) => FileCacheThread (fcacher) => in_filter : AVThread (avhread)
    
    // InfoFrameFilter decoded_info("decoded", &gl_in_filter);
    DummyFrameFilter decoded_info("decoded", true);
    AVThread         avthread1("avthread1", decoded_info);
    AVThread         avthread2("avthread2", decoded_info);
    AVThread         avthread3("avthread3", decoded_info);
    
    // FifoFrameFilter  &in_filter = avthread.getFrameFilter(); // request framefilter from AVThread
    // InfoFrameFilter in_filter("encoded");
    
    int n_blocks = 10;
    int i;
    
    // create ValkkaFS and WriterThread.  15 frames per block, 50 blocks
    ValkkaFS fs("/dev/sda1", "/home/sampsa/python3_packages/valkka_examples/api_level_2/qt/fs_directory/blockfile", 2097152, n_blocks, false); // dumpfile, blockfile, blocksize, number of blocks, clear blocktable
    FileCacheThread fcacher("cacher");
    ValkkaFSReaderThread freader("reader", fs, fcacher.getFrameFilter()); // freader => fcacher
    
    avthread1.startCall();
    avthread2.startCall();
    avthread3.startCall();
    
    fcacher.startCall();
    freader.startCall();
    
    avthread1.decodingOnCall();
    avthread2.decodingOnCall();
    avthread3.decodingOnCall();
    
    freader.setSlotIdCall(1, 1); // map slot 1 from id 1
    freader.setSlotIdCall(2, 2); // map slot 2 from id 2
    freader.setSlotIdCall(3, 3); // map slot 3 from id 3
    
    FileStreamContext ctx1(1, &avthread1.getFrameFilter()); // slot, framefilter
    FileStreamContext ctx2(2, &avthread2.getFrameFilter()); // slot, framefilter
    FileStreamContext ctx3(3, &avthread3.getFrameFilter()); // slot, framefilter
    
    fcacher.registerStreamCall(ctx1);
    fcacher.registerStreamCall(ctx2);
    fcacher.registerStreamCall(ctx3);
    
    ValkkaFSTool fstool(fs);
    
    for(i=0;i<=n_blocks;i++) {
        fstool.dumpBlock(i);
    }
    
    /*
    std::cout << "\nSEEK\n" << std::endl;
    fcacher.seekStreamsCall(1554397742533); // set the target time.  There are no frames, so nothing happens
    sleep_for(0.1s);
    */
    
    std::list<std::size_t> block_list = {0, 1, 2, 3, 4, 5};
    freader.pullBlocksCall(block_list); // send frames.  Once the transmission ends, seek iteration is performed
    
    // now the new frames are available
    sleep_for(2s);
    
    // WARNING: remember to seek to a time that's withing your requested blocks!
    long int mstime;
    mstime = 1554397786825; // just after key-frames @ block 3
    
    fcacher.seekStreamsCall(mstime); // set the target time
    sleep_for(4s);
    
    freader.stopCall();
    fcacher.stopCall();
    
    avthread1.stopCall();
    avthread2.stopCall();
    avthread3.stopCall();
}


void test_3() {  
  const char* name = "@TEST: cachestream_decode_test: test 3: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
  
  
  
  
  
  
}


void test_4() {
  
  const char* name = "@TEST: cachestream_decode_test: test 4: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}


void test_5() {
  
  const char* name = "@TEST: cachestream_decode_test: test 5: ";
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


