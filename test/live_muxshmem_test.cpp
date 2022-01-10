/*
 * live_muxshmem_test.cpp :
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
 *  @file    live_muxshmem_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.3.0 
 *  
 *  @brief 
 *
 */ 

#include "framefifo.h"
#include "framefilter.h"
#include "logging.h"
#include "avdep.h"
#include "livethread.h"
#include "muxshmem.h"
#include "muxer.h"

#include "test_import.h" // don't forget this


using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
    const char* name = "@TEST: live_muxshmem_test: test 1: ";
    std::cout << name <<"** @@Write FragMP4 to shmem ringbuffer.  10 byte cells to test overflow. **" << std::endl;

    // (LiveThread:livethread) --> {FrameFilter:info} --> {FragMP4MuxFrameFilter:fragmp4muxer} --> {FragMP4ShmemFrameFilter:fragmp4shmem}
    // FragMP4ShmemFrameFilter fragmp4_filter("fragmp4shmem", 10, 1024*1024*10, 1000);
    FragMP4ShmemFrameFilter fragmp4_filter("fragmp4shmem", 10, 10, 1000); // test buffer overflow
    FragMP4MuxFrameFilter fragmp4_muxer("fragmp4muxer", &fragmp4_filter); 
    InfoFrameFilter info("info", &fragmp4_muxer);
    LiveThread livethread("live");

    fragmp4_muxer.activate(); // don't forget!

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    std::cout << name << "starting threads" << std::endl;
    livethread.startCall();

    std::cout << name << "registering stream" << std::endl;
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &info); // Request livethread to write into filter info
    livethread.registerStreamCall(ctx);

    // sleep_for(1s);

    std::cout << name << "playing stream !" << std::endl;
    livethread.playStreamCall(ctx);

    sleep_for(5s);
    // sleep_for(604800s); //one week

    std::cout << name << "stopping threads" << std::endl;
    livethread.stopCall();
    // avthread.  stopCall();
}


void test_2() {
    const char* name = "@TEST: live_muxshmem_test: test 2: ";
    std::cout << name <<"** @@Server & client test for FragMP4 sharedmem **" << std::endl;
    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }
    std::cout << name << "** test rtsp stream 1: " << stream_1 << std::endl;

    // EventFd efd = EventFd(); // create the event file descriptor before fork
    int nmax = 1024*1024*3; // 3 MB
    int nqueue = 50;
    bool ok;

    int pid = fork();

    if (pid > 0) // in parent
    {
        std::cout << "Parent starting fragmp4 shmem server" << std::endl;
        FragMP4ShmemFrameFilter fragmp4_filter("fragmp4", nqueue, nmax, 1000); // test buffer overflow
        FragMP4MuxFrameFilter fragmp4_muxer("fragmp4muxer", &fragmp4_filter); 
        InfoFrameFilter info("info", &fragmp4_muxer);
        LiveThread livethread("live");

        fragmp4_muxer.activate(); // don't forget!

        std::cout << name << "starting threads" << std::endl;
        livethread.startCall();

        std::cout << name << "registering stream" << std::endl;
        LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &info); // Request livethread to write into filter info
        livethread.registerStreamCall(ctx);

        // sleep_for(1s);

        std::cout << name << "playing stream !" << std::endl;
        livethread.playStreamCall(ctx);

        sleep_for(3s);
        // sleep_for(604800s); //one week

        std::cout << name << "deactivate" << std::endl;
        fragmp4_muxer.deActivate();
        sleep_for(3s);

        std::cout << name << "reactivate" << std::endl;
        fragmp4_muxer.activate();
        sleep_for(3s);

        std::cout << name << "stopping threads" << std::endl;
        livethread.stopCall();
        // avthread.  stopCall();
        fragmp4_muxer.deActivate();
    }
    else
    {
        sleep_for(0.1s); // a smaller delay.. so that we don't miss the ftyp & moov packets due to overflow
        std::cout << "Client reading shmem ringbuffer" << std::endl;
        // must use this in order to work with valgrind use 2+ secs
        // .. otherwise valgrind does some strange shit!  It doesn't reserve memory for the client: 
        FragMP4SharedMemRingBuffer rb("fragmp4", nqueue, nmax, false); // client-side shmem ring-buffer
        int index, n, ii, deadcount;
        FragMP4Meta meta;
        deadcount=0;
        while (true) 
        {
            ok = rb.clientPull(index, &meta);
            if (ok)
            {
                n = meta.size;
                std::cout << "CLIENT: cell index " << index << " has " << n << " bytes" << std::endl;
                for (ii = 0; ii < std::min(n, 10); ii++)
                {
                    std::cout << int(rb.shmems[index]->payload[ii]) << " ";
                }
                std::cout << std::endl;
                std::cout << "CLIENT: slot : " << meta.slot  << std::endl;
                std::cout << "CLIENT: name : " << meta.name  << std::endl;
                std::cout << "CLIENT: first: " << meta.is_first << std::endl;
            }
            else
            {
                std::cout << "CLIENT: semaphore timed out!" << std::endl;
                std::cout << std::endl;
                deadcount++;
            }
            if (deadcount >= 10)
            {
                break;
            }
        }
        std::cout << "CLIENT: exit" << std::endl;
    }
}


void test_3() {
  
  const char* name = "@TEST: live_muxshmem_test: test 3: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}


void test_4() {
  
  const char* name = "@TEST: live_muxshmem_test: test 4: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}


void test_5() {
  
  const char* name = "@TEST: live_muxshmem_test: test 5: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}



int main(int argc, char** argcv) {
  if (argc<2) {
    std::cout << argcv[0] << " needs an integer argument.  Second interger argument (optional) is verbosity" << std::endl;
  }
  else {
    ffmpeg_av_register_all(); // never forget!

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


