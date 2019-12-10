/*
 * valkkafs_test.cpp :
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
 *  @file    valkkafs_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.15.0 
 *  
 *  @brief 
 *
 */ 

#include "framefifo.h"
#include "framefilter.h"
#include "logging.h"
#include "avdep.h"
#include "valkkafs.h"
#include "test_import.h"


using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
    const char* name = "@TEST: valkkafs_test: test 1: ";
    std::cout << name <<"** @@Write and recover ValkkaFS info **" << std::endl;

    ValkkaFS fs("disk.dat", "block.dat", 1024*1024, 10, true);

    fs.setVal(1,1,22);
    
    fs.dumpTable();
    
    fs.readTable();

    std::cout << fs.getVal(1,1) << std::endl;
    
    ValkkaFS fs2("disk.dat", "block.dat", 1024*1024, 10, false);
    // fs2.read();
    
    std::cout << fs2.getVal(1,1) << std::endl;
    
}


void test_2() {
    const char* name = "@TEST: valkkafs_test: test 2: ";
    std::cout << name <<"** @@Write ValkkaFS info **" << std::endl;
  
    ValkkaFS fs("disk.dat", "block.dat", 1024*1024, 10, false);

    std::cout << fs.getDevice() << std::endl;
    std::cout << fs.getDeviceSize() << std::endl;
    
    fs.clearDevice(true, true); // write through, be verbose
    fs.clearDevice(false, true); // just stripe, be verbose
    
    fs.clearTable();
    fs.dumpTable();
}


void test_3() {
    const char* name = "@TEST: valkkafs_test: test 3: ";
    std::cout << name <<"** @@Test Frame serialization **" << std::endl;

    BasicFrame *f = new BasicFrame();
    f->resize(1024*1024);
    f->subsession_index=1;
    std::fill(f->payload.begin(), f->payload.end(), 12);
    f->mstimestamp = 1001;

    // std::fstream os("framedump", std::fstream::binary | std::fstream::out);
    RaWriter os("framedump");
    f->dump(123, os);
    os.close_();
   
    std::cout << "frame:" << *f << std::endl;
    
    BasicFrame *f2 = new BasicFrame();
    
    // std::fstream is("framedump", std::ios::binary | std::fstream::in);
    RawReader is("framedump");
    f2->read(is);
    is.close_();
    
    std::cout << "frame 2:" << *f2 << std::endl;
    
    delete f;
    delete f2;
}


void test_4() {
    const char* name = "@TEST: valkkafs_test: test 4: ";
    std::cout << name <<"** @@Test frame serialization 2 **" << std::endl;
    int i;
    
    BasicFrame *f = new BasicFrame();
    f->resize(1024*1024);
    // f->resize(10);
    
    // std::fstream os("framedump", std::fstream::binary | std::fstream::out);
    RaWriter os("framedump");
    for(i=0;i<10;i++) {
        f->subsession_index=i%2;
        f->n_slot=i;
        std::fill(f->payload.begin(), f->payload.end(), i+1);
        f->mstimestamp = 100+i;
        f->dump(3, os); // device_id, writer
        std::cout << *f << std::endl;
    }
    os.close_();
    
    std::cout << std::endl;
    
    // return;
    
    std::size_t id;
    // std::fstream is("framedump", std::fstream::binary | std::fstream::in);
    RawReader is("framedump");
    for(i=0;i<10;i++) {
        id=f->read(is);
        std::cout << id << ": " << *f << std::endl;
    }
    is.close_();
    
    delete f;
}


void test_5() {
    const char* name = "@TEST: valkkafs_test: test 5: ";
    std::cout << name <<"** @@Test ValkkaFSWriterThread **" << std::endl;

    BasicFrame *f = new BasicFrame();
    f->resize(1024*1024);
    f->subsession_index=0; // at least one stream must be the "lead stream" and have subsession_index = 0
    f->n_slot=1;
    std::fill(f->payload.begin(), f->payload.end(), 1);
    f->mstimestamp = 100;

    ValkkaFS fs("disk.dat", "block.dat", 20*1024*1024, 10, true); // last parameter: init blocktable
    ValkkaFSWriterThread ft("writer", fs);
    
    FrameFilter &filt = ft.getFrameFilter();
    
    ft.startCall();
    sleep_for(1s);
    ft.setSlotIdCall(1, 1); // slot, id
    
    int i;
    for(i=0;i<10;i++) {
        std::cout << "sending " << i << std::endl;
        filt.run(f);
    }
    
    std::cout << "sleepin'" << std::endl;
    sleep_for(5s);
    
    ft.stopCall();
    
    delete f;
}


void test_6() {
    const char* name = "@TEST: valkkafs_test: test 6: ";
    std::cout << name <<"** @@Test ValkkaFSWriterThread 2 **" << std::endl;

    // construct a dummy frame
    BasicFrame *f = new BasicFrame();
    f->resize(1024*1024);
    f->subsession_index=0;
    f->n_slot=1;
    std::fill(f->payload.begin(), f->payload.end(), 1);
    f->mstimestamp = 100;

    // create ValkkaFS and WriterThread
    ValkkaFS fs("disk.dat", "block.dat", 10*1024*1024, 10, true); // blocksize, number of blocks .. init blocktable
    ValkkaFSWriterThread ft("writer", fs);
    
    FrameFilter &filt = ft.getFrameFilter();
    
    ft.startCall();
    
    sleep_for(1s);
    std::cout << "\nSetting slot" << std::endl;
    ft.setSlotIdCall(1, 123);ft.reportSlotIdCall();
    std::cout << "\nSetting slot - same again" << std::endl;
    ft.setSlotIdCall(1, 123);ft.reportSlotIdCall();
    std::cout << "\nSetting slot - new" << std::endl;
    ft.setSlotIdCall(3, 123);ft.reportSlotIdCall();
    std::cout << "\nUnsetting slot" << std::endl;
    ft.unSetSlotIdCall(1);ft.reportSlotIdCall();
    std::cout << "\nClear all" << std::endl;
    ft.clearSlotIdCall();ft.reportSlotIdCall();
    
    std::cout << "\nWriting frames" << std::endl;
    int i;
    for(i=0;i<10;i++) {
        filt.run(f);
    }
    
    sleep_for(5s);
    
    std::cout << "\nStopping" << std::endl;
    ft.stopCall();
    
    delete f;
}


void test_7() {
    const char* name = "@TEST: valkkafs_test: test 2: ";
    std::cout << name <<"** @@Write ValkkaFS info **" << std::endl;
    return; // WARNING: dangerous test, as it will wipe out /dev/sda
  
    ValkkaFS fs("/dev/sda", "block.dat", 20*1024*1024, 3, false);
    fs.clearDevice(false, true); // just stripe, be verbose
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
      case(7):
        test_7();
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


