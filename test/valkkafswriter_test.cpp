/*
 * valkkafswriter_test.cpp :
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
 *  @file    valkkafswriter_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.17.4 
 *  
 *  @brief 
 *
 */ 

#include "framefifo.h"
#include "framefilter.h"
#include "fileframefilter.h"
#include "logging.h"
#include "avdep.h"
#include "valkkafs.h"
#include "valkkafsreader.h"
#include "test_import.h"


using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
    const char* name = "@TEST: valkkafswriter_test: test 1: ";
    std::cout << name <<"** @@Write frames **" << std::endl;
  
    // construct a dummy frame
    BasicFrame *f = new BasicFrame();
    f->resize(1024);
    f->subsession_index=0;
    f->n_slot=1;
    f->media_type =AVMEDIA_TYPE_VIDEO;
    f->codec_id   =AV_CODEC_ID_H264;
    
    // create ValkkaFS and WriterThread
    ValkkaFS fs("disk.dump", "block.dat", 10*1024, 10, true); // dumpfile, blockfile, blocksize, number of blocks, init blocktable
    // 10 blocks
    // so, with this frame size, its 9 frames per block
    
    std::cout << "ValkkaFS blocksize: " << fs.getBlockSize() << std::endl;
    // return;
    
    ValkkaFSWriterThread ft("writer", fs);

    fs.clearDevice(); // create and fill device file "disk.dump"
    fs.clearTable(); // clears blocktable and dumps it to disk
    // fs.dumpTable();
    
    FrameFilter &filt = ft.getFrameFilter();

    ft.startCall();

    sleep_for(1s);
    ft.setSlotIdCall(1, 123);
    
    std::cout << "\nWriting frames" << std::endl;
    int i, j;
    int nb=7; // write this many blocks // 6, 7
    int bf=4; // frames per block
    int count = 0;
    
    for(j=0;j<=nb; j++) {
        std::cout << "\nBLOCK " << j << "\n";
        for(i=0;i<=4;i++) {
            std::fill(f->payload.begin(), f->payload.end(), 0);
            // let's simulate a H264 frame
            std::copy(nalstamp.begin(),nalstamp.end(),f->payload.begin()); // insert 0001
            long int mstimestamp = 1000*(j+1)+i*100;
            f->mstimestamp = mstimestamp;
            if (i%5==0) { // every 5.th frame a key frame
                std::cout << "marking key" << std::endl;
                f->payload[nalstamp.size()]=(31 & 7); // mark fake sps frame
            }
            // std::cout << "payload: " << f->dumpPayload() << std::endl;
            // std::cout << "isSeekable: " << f->isSeekable() << std::endl;
            
            count += f->calcSize();
            std::cout << i << ": framesize " << f->calcSize() << " " << count << std::endl;
            
            filt.run(f);
            sleep_for(0.01s); // otherwise frames will overflow ..
        }
    }

    sleep_for(1s);

    std::cout << "\nStopping" << std::endl;
    ft.stopCall();
    delete f;
    
    // return;
    
    fs.reportTable();
    
    fs.dumpTable(); // to disk
    
    ValkkaFSTool fstool(fs);
    
    for(i=0;i<=nb;i++) {
        fstool.dumpBlock(i);
    }
}


void test_2() {
    const char* name = "@TEST: valkkafswriter_test: test 2: ";
    std::cout << name <<"** @@Read frames with ValkkaFSTool.  Continuation of test 1. **" << std::endl;
    
    ValkkaFS fs("disk.dump", "block.dat", 10*1024, 10, false); // dumpfile, blockfile, blocksize, number of blocks, don't overwrite blocktable
    // fs.read();
    
    fs.reportTable();
    
    ValkkaFSTool fstool(fs);
        
    std::size_t i;
    for(i=0;i<=fs.get_n_blocks();i++) {
        // std::cout << "i = " << i << std::endl;
        fstool.dumpBlock(i);
    }
  
}


void test_3() { // don't do: 3, 2 => crash .. of course .. number of blocks don't match
    const char* name = "@TEST: valkkafswriter_test: test 3: ";
    std::cout << name <<"** @@Write frames with filesystem wrap **" << std::endl;

    // construct a dummy frame
    BasicFrame *f = new BasicFrame();
    f->resize(1024);
    f->subsession_index=0;
    f->n_slot=1;
    f->media_type =AVMEDIA_TYPE_VIDEO;
    f->codec_id   =AV_CODEC_ID_H264;
    
    // create ValkkaFS and WriterThread
    ValkkaFS fs("disk.dump", "block.dat", 10*1024, 5, true); // dumpfile, blockfile, blocksize, number of blocks, clear blocktable
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
    for(i=0;i<=(10*nb);i++) { // write over the block
        // let's simulate a H264 frame
        std::fill(f->payload.begin(), f->payload.end(), 0);
        std::copy(nalstamp.begin(),nalstamp.end(),f->payload.begin()); // insert 0001
        long int mstimestamp = 1000*(i+1);
        f->mstimestamp = mstimestamp;
        if (i%5==0) { // every 5.th frame a key frame
            f->payload[4]=(31 & 7); // mark fake sps frame
        }
        filt.run(f);
        sleep_for(0.01s); // otherwise frames will overflow ..
    }

    sleep_for(1s);

    std::cout << "\nStopping" << std::endl;
    ft.stopCall();
    delete f;
    
    fs.reportTable(0, 0, true);
    
    fs.dumpTable(); // to disk
    
    ValkkaFSTool fstool(fs);
    
    for(i=0;i<=nb;i++) {
        fstool.dumpBlock(i);
    }
}


void test_4() { // 3, 4
    const char* name = "@TEST: valkkafswriter_test: test 4: ";
    std::cout << name <<"** @@Use valkkafsreader to read frames from ValkkaFS.  Pass them to InitStreamFrameFilter. **" << std::endl;
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
    
    InfoFrameFilter info_filter("info");
    InitStreamFrameFilter init_filter("init", &info_filter);
    ValkkaFSReaderThread ft("reader", fs, init_filter);
    
    std::list<std::size_t> block_list = {0, 1, 2};
    
    ft.startCall();

    sleep_for(1s);
    
    ft.setSlotIdCall(2, 123);
    ft.reportSlotIdCall();
    
    ft.pullBlocksCall(block_list);
    
    sleep_for(2s);
    
    ft.stopCall();
}



void test_5() {
    const char* name = "@TEST: valkkafswriter_test: test 5: ";
    std::cout << name <<"** @@Read frames with ValkkaFSTool **" << std::endl;
    return;
    
    ValkkaFS fs("/dev/sda1", "/home/sampsa/python3_packages/valkka_examples/api_level_2/qt/fs_directory/blockfile", 2097152, 100, false); // dumpfile, blockfile, blocksize, number of blocks, don't overwrite blocktable
    // fs.read();
    
    fs.reportTable();
    
    ValkkaFSTool fstool(fs);
        
    std::size_t i;
    for(i=0;i<=fs.get_n_blocks();i++) {
        fstool.dumpBlock(i);
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


