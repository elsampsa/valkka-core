/*
 * cachestream_test.cpp : testing dumping frames from valkkafs into cachestream
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
 *  @file    cachestream_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.17.4 
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
#include "test_import.h"


using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


/** Testing callbacks from a FileCacheThread
 * 
 * - FileCacheThread calls this to inform that it is missing frames at mstimestamp
 * - That call goes to python, but here we are testing in cpp
 * - In this test class, pass the constructir the list of blocks that are passed to FileCacheThread when it triggers the callback
 */
std::list<std::size_t> *BLOCK_LIST;
ValkkaFSReaderThread *FREADER;
void FUNC(long int mstimestamp) {
    std::cout << "callback : " << mstimestamp << std::endl;
    if (FREADER) {
        FREADER->pullBlocksCall(*BLOCK_LIST);
    }
}



void test_1() {
    const char* name = "@TEST: cachestream_test: test 1: ";
    std::cout << name <<"** @@Test FileCacheThread **" << std::endl;
  
    InfoFrameFilter info("info");
    
    FileCacheThread fcacher("cacher");
    
    fcacher.startCall();
    
    sleep_for(1s);
        
    FileStreamContext ctx(2, &info); // slot, framefilter
    fcacher.registerStreamCall(ctx);
    sleep_for(1s);
    fcacher.deregisterStreamCall(ctx);
    sleep_for(1s);
    fcacher.seekStreamsCall(9000);
    fcacher.playStreamsCall();
    fcacher.stopStreamsCall();
    sleep_for(1s);
    fcacher.stopCall();
}


void test_2() { // run first valkkafswriter_test 3
    // [ValkkaFSReaderThread] -->> [FileCacheThread] 
    
    const char* name = "@TEST: cachestream_test: test 1: ";
    std::cout << name <<"** @@Dump cached frames from FileCacheThread::cache_stream **" << std::endl;
  
    int i;
    // create ValkkaFS and WriterReaderThread
    ValkkaFS fs("disk.dump", "block.dat", 10*1024, 5, false); // dumpfile, blockfile, blocksize, number of blocks, clear blocktable
    // fs.read();
    // 5 blocks
    // so, with this frame size, its 9 frames per block
    
    fs.reportTable(0, 0, true);
    ValkkaFSTool fstool(fs);
    for(i=0;i<fs.get_n_blocks();i++) {
        // std::cout << i << std::endl;
        fstool.dumpBlock(i);
    }
    
    InfoFrameFilter info("info");
    
    FileCacheThread fcacher("cacher");
    ValkkaFSReaderThread freader("reader", fs, fcacher.getFrameFilter());
    // ValkkaFSReaderThread freader("reader", fs, info);
    
    fcacher.startCall();
    freader.startCall();
    
    sleep_for(1s);
    
    freader.setSlotIdCall(2, 123);
    freader.reportSlotIdCall();
    
    FileStreamContext ctx(2, &info); // slot, framefilter
    fcacher.registerStreamCall(ctx);
    
    std::cout << "\npull1\n";
    std::list<std::size_t> block_list = {0};
    freader.pullBlocksCall(block_list);
    sleep_for(1s);
    // ok, by now it should be in the cache
    fcacher.dumpCache();
    
    std::cout << "\npull2\n";
    block_list = {99};
    freader.pullBlocksCall(block_list);
    sleep_for(1s);
    // ok, by now it should be in the cache
    fcacher.dumpCache();
    
    std::cout << "\npull3\n";
    block_list = {0,1};
    freader.pullBlocksCall(block_list);
    sleep_for(1s);
    // ok, by now it should be in the cache
    fcacher.dumpCache();
    
    freader.stopCall();
    fcacher.stopCall();
}



void test_3() {
    /*

    A) A standalone test

    1) Write some blocks

    2) Seek: => OK

    - Sending new blocks and seek are separate operations
    - SeekCall simply sets the target time
    - .. if the requested target time is within limits of the current blocks, then start seek iteration
    - Finished receiving new blocks triggers always the seek iteration to the current target time
    
    - Seek call to a python process
    - .. that inspects the blocktable : is it necessary to pull more blocks before seek
    - forward the seek to the cpp thread
    
    - So the python process acts as the "gatekeeper" for seeks.
    - Consider also this : file:///home/sampsa/C/valkka/docs/html/valkkafs.html
    - The first block to be sent is the block for which target_time >= k-max row
    
    B) Next tests:

    - Same, but also play => OK
    - Distance (not time, but number of frames) from the block limit should be monitored .. and issue automatic requests for more frames
    - .. in fact, there should be a callback to python that informs about the current time .. and if the time is too close to the block limit, then send more blocks
    - TODO: fix the python callback and emit times
    - TODO: check that playing over the frame limit does not crash
    
    C) A standalone test with large gaps in recorded frame timestamps

    - TODO: think about this ..
    */


    const char* name = "@TEST: cachestream_test: test 3: ";
    std::cout << name <<"** @@Write and seek **" << std::endl;

    std::size_t payload_size = 20; // there is more to the frame than just the payload
    std::size_t frames_per_block = 15;
    std::size_t n_blocks = 50;
    
    // construct a dummy frame
    BasicFrame *f = new BasicFrame();
    f->resize(payload_size);
    f->subsession_index=0;
    f->n_slot=1;
    f->media_type =AVMEDIA_TYPE_VIDEO;
    f->codec_id   =AV_CODEC_ID_H264;
    
    // create ValkkaFS and WriterThread.  15 frames per block, 50 blocks
    ValkkaFS fs("disk.dump", "block.dat", payload_size*frames_per_block, n_blocks, true); // dumpfile, blockfile, blocksize, number of blocks, clear blocktable
    // 5 blocks
    // so, with this frame size, its 9 frames per block
    ValkkaFSWriterThread ft("writer", fs);
    
    fs.clearDevice(); // create and fill device file "disk.dump"
    fs.clearTable();  // clears blocktable and dumps it to disk
    
    FrameFilter &filt = ft.getFrameFilter();

    ft.startCall();

    sleep_for(1s);
    ft.setSlotIdCall(1, 123);
    
    std::cout << "\nWriting frames" << std::endl;
    int i;
    int nb=7; // write this many blocks
    for(i=0;i<=(nb*int(frames_per_block));i++) { // write over the block
        // let's simulate a H264 frame
        std::fill(f->payload.begin(), f->payload.end(), 0);
        std::copy(nalstamp.begin(),nalstamp.end(),f->payload.begin()); // insert 0001
        long int mstimestamp = 1000*(i+1);
        f->mstimestamp = mstimestamp;
        if (i%4==0) { // every 4.th frame a key frame
            f->payload[4]=(31 & 7); // mark fake sps frame
        }
        filt.run(f);
        sleep_for(0.01s); // otherwise fifo will overflow ..
    }
    sleep_for(1s);

    // 1 000 => 106 000
    
    std::cout << "\nStopping" << std::endl;
    ft.stopCall();
    delete f;
    
    fs.reportTable(0, 0, true);
    
    fs.dumpTable(); // to disk
    
    ValkkaFSTool fstool(fs);
    
    for(i=0;i<=n_blocks;i++) {
        fstool.dumpBlock(i);
    }
    
    InfoFrameFilter info("info");
    FileCacheThread fcacher("cacher");
    ValkkaFSReaderThread freader("reader", fs, fcacher.getFrameFilter());
    
    // prepare the callback function
    void (*fpointer)(long int) = &FUNC;
    std::list<std::size_t> block_list = {0, 1};
    BLOCK_LIST = &block_list;
    FREADER = &freader;
    
    // fcacher.setCallback(fpointer);
    
    fcacher.startCall();
    freader.startCall();
    
    sleep_for(1s);
    
    freader.setSlotIdCall(2, 123); // map id number 123 to slot number 2
    // freader.reportSlotIdCall();
    
    FileStreamContext ctx(2, &info); // slot, framefilter : write frames of slot 2 to framefilter info
    fcacher.registerStreamCall(ctx);
    
    fcacher.seekStreamsCall(7900); // set the target time.  There are no frames, so nothing happens
    sleep_for(0.1s);
    freader.pullBlocksCall(block_list); // send frames.  Once the transmission ends, seek iteration is performed
    
    sleep_for(2s);
    
    std::cout << "\nSEEK TO 60000\n" << std::endl;
    fcacher.seekStreamsCall(60000); // set the target time.  No frames for this time stamp, so nothing happens
    
    sleep_for(2s);
    
    block_list = {11, 12, 13, 14, 15};
    freader.pullBlocksCall(block_list); // send frames.  After the transmission, seek iteration proceeds
    
    sleep_for(2s);
    
    std::cout << "\nSEEK TO 64500\n" << std::endl;
    fcacher.seekStreamsCall(64500); // set the target time.  Frames obtained in the last transmission are OK, so seek happens
    
    fcacher.stopCall();
    freader.stopCall();
}


void test_4() {
  const char* name = "@TEST: cachestream_test: test 4: ";
    std::cout << name <<"** @@Write, seek and play **" << std::endl;

    std::size_t payload_size = 20; // there is more to the frame than just the payload
    std::size_t frames_per_block = 15; // 15 secs per block
    std::size_t n_blocks = 50;
    
    // construct a dummy frame
    BasicFrame *f = new BasicFrame();
    f->resize(payload_size);
    f->subsession_index=0;
    f->n_slot=1;
    f->media_type =AVMEDIA_TYPE_VIDEO;
    f->codec_id   =AV_CODEC_ID_H264;
    
    // create ValkkaFS and WriterThread.  15 frames per block, 50 blocks
    ValkkaFS fs("disk.dump", "block.dat", payload_size*frames_per_block, n_blocks, true); // dumpfile, blockfile, blocksize, number of blocks, clear blocktable
    // 5 blocks
    // so, with this frame size, its 9 frames per block
    ValkkaFSWriterThread ft("writer", fs);
    
    fs.clearDevice(); // create and fill device file "disk.dump"
    fs.clearTable();  // clears blocktable and dumps it to disk
    
    FrameFilter &filt = ft.getFrameFilter();

    ft.startCall();

    sleep_for(1s);
    ft.setSlotIdCall(1, 123);
    
    std::cout << "\nWriting frames" << std::endl;
    int i;
    int nb=7; // write this many blocks
    for(i=0;i<=(nb*int(frames_per_block));i++) { // write over the block
        // let's simulate a H264 frame
        std::fill(f->payload.begin(), f->payload.end(), 0);
        std::copy(nalstamp.begin(),nalstamp.end(),f->payload.begin()); // insert 0001
        long int mstimestamp = 1000*(i+1); 
        f->mstimestamp = mstimestamp;
        if (i%4==0) { // every 4.th frame a key frame
            f->payload[4]=(31 & 7); // mark fake sps frame
        }
        filt.run(f);
        sleep_for(0.01s); // otherwise fifo will overflow ..
    }
    sleep_for(1s);

    
    
    // 1 000 => 106 000
    
    std::cout << "\nStopping" << std::endl;
    ft.stopCall();
    delete f;
    
    fs.reportTable(0, 0, true);
    
    fs.dumpTable(); // to disk
    
    ValkkaFSTool fstool(fs);
    
    for(i=0;i<=n_blocks;i++) {
        fstool.dumpBlock(i);
    }
    
    InfoFrameFilter info("info");
    FileCacheThread fcacher("cacher");
    ValkkaFSReaderThread freader("reader", fs, fcacher.getFrameFilter());
    
    std::list<std::size_t> block_list = {0, 1, 2, 3, 4, 5};
        
    fcacher.startCall();
    freader.startCall();
    
    sleep_for(1s);
    
    freader.setSlotIdCall(2, 123); // map id number 123 to slot number 2
    // freader.reportSlotIdCall();
    
    FileStreamContext ctx(2, &info); // slot, framefilter : write frames of slot 2 to framefilter info
    fcacher.registerStreamCall(ctx);
    
    std::cout << "\nSEEK\n" << std::endl;
    fcacher.seekStreamsCall(7900); // set the target time.  There are no frames, so nothing happens
    sleep_for(0.1s);
    freader.pullBlocksCall(block_list); // send frames.  Once the transmission ends, seek iteration is performed
    sleep_for(2s);
    
    /*
    std::cout << "\nPLAY\n" << std::endl;
    fcacher.playStreamsCall();
    sleep_for(3s);
    std::cout << "\nSTOP\n" << std::endl;
    fcacher.stopStreamsCall();
    sleep_for(3s);
    std::cout << "\nPLAY\n" << std::endl;
    fcacher.playStreamsCall();
    sleep_for(3s);
    */
    
    std::cout << "\nSEEK AND PLAY OVER EDGE\n" << std::endl;
    fcacher.seekStreamsCall(26000); // set the target time.  There are no frames, so nothing happens
    sleep_for(2s);
    fcacher.playStreamsCall();
    sleep_for(10s);
    std::cout << "\nSTOP\n" << std::endl;
    fcacher.stopStreamsCall();
    sleep_for(3s);
    
    fcacher.stopCall();
    freader.stopCall();
}


void test_5() {
  
  const char* name = "@TEST: cachestream_test: test 5: ";
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


