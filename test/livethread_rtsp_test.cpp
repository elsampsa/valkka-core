/*
 * livethread_rtsp_test.cpp : test the rtsp server
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
 *  @file    livethread_rtsp_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2018
 *  @version 1.3.3 
 *  
 *  @brief 
 *
 */ 

#include "livethread.h"
#include "framefilter.h"
#include "logging.h"
#include "avdep.h"
#include "test_import.h"



using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");

int portnum = 8555;


void test_1() {
  const char* name = "@TEST: livethread_rtsp_test: test 1: ";
  std::cout << name <<"** @@Test rtsp server**" << std::endl;
  
  if (!stream_1) {
    std::cout << name <<"ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1"<< std::endl;
    exit(2);
  }
  std::cout << name <<"** test rtsp stream 1: "<< stream_1 << std::endl;
  
  setLiveOutPacketBuffermaxSize(300*1024); // 300 kB
  
  LiveThread livethread2("livethread2");
  livethread2.setRTSPServer(portnum);
  FifoFrameFilter& fifo_filter =livethread2.getFrameFilter();
  
  BriefInfoFrameFilter info_filter("info_filter",&fifo_filter);
  // CountFrameFilter info_filter("info_filter",&fifo_filter);
  
  std::cout << "starting live threads" << std::endl;
  livethread2.startCall();
  
  sleep_for(1s);

  LiveOutboundContext out_ctx = LiveOutboundContext(LiveConnectionType::rtsp, std::string("kokkelis"), 2, 0); 
  livethread2.registerOutboundCall(out_ctx);
  
  sleep_for(1s);
  
  std::cout << "stopping live thread" << std::endl;
  livethread2.stopCall();  
}


void test_2() {
  const char* name = "@TEST: livethread_rtsp_test: test 2: ";
  std::cout << name <<"** @@Sending frames from client to rtsp server**" << std::endl;
  
  if (!stream_1) {
    std::cout << name <<"ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1"<< std::endl;
    exit(2);
  }
  std::cout << name <<"** test rtsp stream 1: "<< stream_1 << std::endl;
  
  setLiveOutPacketBuffermaxSize(300*1024); // 300 kB
  
  // filtergraph:
  // (LiveThread:livethread) --> {BriefInfoFrameFilter:info_filter) --> (FifoFrameFilter: fifo_filter) -->> (LiveThread:livethread2) 
  LiveThread  livethread("livethread");
  LiveThread  livethread2("livethread"); // stack size for incoming fifo
  livethread2.setRTSPServer(portnum);
  
  FifoFrameFilter& fifo_filter =livethread2.getFrameFilter();
  
  // BriefInfoFrameFilter info_filter("info_filter",&fifo_filter);
  // CountFrameFilter info_filter("info_filter",&fifo_filter);
  
  std::cout << "starting live threads" << std::endl;
  livethread. startCall();
  livethread2.startCall();
  
  sleep_for(1s);

  LiveOutboundContext out_ctx = LiveOutboundContext(LiveConnectionType::rtsp, std::string("kokkelis"), 2, 50000); // ffplay rtsp://localhost:8554/kokkelis  .. that last "5000" is not used by the rtsp server
  livethread2.registerOutboundCall(out_ctx);
  
  sleep_for(1s);
  
  // LiveConnectionContext ctx =LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &info_filter);
  LiveConnectionContext ctx =LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &fifo_filter);
  livethread.registerStreamCall(ctx);
  livethread.playStreamCall(ctx);
  
  sleep_for(2s);
  
  livethread.stopStreamCall(ctx);
  
  sleep_for(1s);
  
  std::cout << "stopping live thread" << std::endl;
  livethread. stopCall();
  livethread2.stopCall();
  
}


void test_3() {
  const char* name = "@TEST: livethread_rtsp_test: test 3: ";
  std::cout << name <<"** @@Sending frames from client to rtsp server**" << std::endl;
  
  if (!stream_1) {
    std::cout << name <<"ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1"<< std::endl;
    exit(2);
  }
  std::cout << name <<"** test rtsp stream 1: "<< stream_1 << std::endl;
  
  std::string stream_address = std::string("rtsp://localhost:")+std::string(std::to_string(portnum))+std::string("/kokkelis");
  
  std::cout << "stream address: " << stream_address << std::endl;
  
  setLiveOutPacketBuffermaxSize(300*1024); // 300 kB
  
  // filtergraph:
  // (LiveThread:livethread) --> {BriefInfoFrameFilter:info_filter) --> (FifoFrameFilter: fifo_filter) -->> (LiveThread:livethread2) 
  LiveThread  livethread("livethread");
  LiveThread  livethread2("livethread"); // stack size for incoming fifo
  
  livethread2.setRTSPServer(portnum);
  
  FifoFrameFilter& fifo_filter =livethread2.getFrameFilter();
  
  // BriefInfoFrameFilter info_filter("info_filter",&fifo_filter);
  // CountFrameFilter info_filter("info_filter",&fifo_filter);
  
  std::cout << "starting live threads" << std::endl;
  livethread. startCall();
  livethread2.startCall();
  
  sleep_for(1s);

  LiveOutboundContext out_ctx = LiveOutboundContext(LiveConnectionType::rtsp, std::string("kokkelis"), 2, 0); // ffplay rtsp://localhost:8554/kokkelis
  livethread2.registerOutboundCall(out_ctx);
  
  sleep_for(1s);
  
  // LiveConnectionContext ctx =LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &info_filter);
  LiveConnectionContext ctx =LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &fifo_filter);
  
  ctx.reordering_time = 2000*1000;
  ctx.time_correction = TimeCorrectionType::none;
  
  livethread.registerStreamCall(ctx);
  livethread.playStreamCall(ctx);
  
  sleep_for(1*60s);
  
  livethread.stopStreamCall(ctx);
  
  sleep_for(1s);
  
  std::cout << "stopping live thread" << std::endl;
  livethread. stopCall();
  livethread2.stopCall();
  
}




void test_4() {  
  const char* name = "@TEST: livethread_rtsp_test: test 4: ";
  std::cout << name <<"** @@Read from the RTSP Server **" << std::endl;
  
  std::string stream_address = std::string("rtsp://localhost:")+std::string(std::to_string(portnum))+std::string("/kokkelis");
  // std::string stream_address = std::string("rtsp://admin:123456@192.168.0.134");
  
  std::cout << "server: " << stream_address << std::endl;
  
  setLiveOutPacketBuffermaxSize(300*1024); // 300 kB
  
  LiveThread  livethread("livethread");
  DummyFrameFilter dummyfilter("dummy");
  
  std::cout << "starting live thread" << std::endl;
  livethread. startCall();
  
  LiveConnectionContext ctx =LiveConnectionContext(LiveConnectionType::rtsp, stream_address, 2, &dummyfilter);
  ctx.reordering_time = 2000*1000;
  ctx.time_correction = TimeCorrectionType::none;
  
  livethread.registerStreamCall(ctx);
  livethread.playStreamCall(ctx);
  sleep_for(1*60s);
  livethread. stopCall();
}


void test_5() {
  
  const char* name = "@TEST: livethread_rtsp_test: test 5: ";
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


