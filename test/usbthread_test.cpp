/*
 * usbthread_test.cpp : Test USB cam classes
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
 *  @file    usbthread_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.3.3 
 *  
 *  @brief   Test USB cam classes
 *
 */ 

#include "framefifo.h"
#include "framefilter.h"
#include "logging.h"
#include "avdep.h"
#include "avthread.h"
#include "openglthread.h"
#include "livethread.h"
#include "usbthread.h"
#include "test_import.h"


using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {  
    const char* name = "@TEST: usbthread_test: test 1: ";
    std::cout << name <<"** @@Test opening a V4L device **" << std::endl;
    
    InfoFrameFilter f("info");

    // v4l2-ctl -d /dev/video2 --list-formats
    
    USBCameraConnectionContext ctx("/dev/video2", 1, &f);
    
    // USBCameraConnectionContext ctx("/dev/video2", 1, &f);
    
    V4LDevice dev = V4LDevice(ctx);
    // V4LDevice dev = V4LDevice("/dev/video1");
    // V4LDevice dev = V4LDevice("/dev/video2");

    dev.open_();
    // dev.paska();
    
    v4l_status status = dev.getStatus();

    if (status == v4l_status::ok_open) {
        std::cout << "OK!" << std::endl;
    }
    
}


void test_2() {
  const char* name = "@TEST: usbthread_test: test 2: ";
  std::cout << name <<"** @@Test USBDeviceThread **" << std::endl;
  
  InfoFrameFilter f("info");
  USBDeviceThread usbt("usbthread");
  
  std::cout << "start" << std::endl;
  usbt.startCall();
  
  sleep_for(1s);
  
  USBCameraConnectionContext ctx("/dev/video2", 1, &f);
  
  //std::cout << "\nregister" << std::endl;
  //usbt.registerCameraStreamCall(ctx);
  
  sleep_for(1s);
  
  std::cout << "\nplay" << std::endl;
  usbt.playCameraStreamCall(ctx);
  
  sleep_for(3s);
  
  std::cout << "\nstop" << std::endl;
  usbt.stopCameraStreamCall(ctx);
  
  sleep_for(3s);
  
  ///*
  std::cout << "\nplay" << std::endl;
  usbt.playCameraStreamCall(ctx);
  
  sleep_for(30s);
  //*/
  
  ///*
  std::cout << "\nreplay" << std::endl;
  usbt.playCameraStreamCall(ctx);
  
  sleep_for(3s);
  //*/
  
  //std::cout << "\nderegister" << std::endl;
  //usbt.deRegisterCameraStreamCall(ctx);
  
  
  std::cout << "\nthread stop" << std::endl;
  usbt.stopCall();
  
}


void test_3() {
  const char* name = "@TEST: usbthread_test: test 3: ";
  std::cout << name <<"** @@Test USBDeviceThread 2 **" << std::endl;
  
  InfoFrameFilter f("info");
  USBDeviceThread usbt("usbthread");
  
  std::cout << "start" << std::endl;
  usbt.startCall();
  
  sleep_for(1s);
  
  USBCameraConnectionContext ctx("/dev/video2", 1, &f);
  
  //std::cout << "\nregister" << std::endl;
  //usbt.registerCameraStreamCall(ctx);
  
  sleep_for(1s);
  
  std::cout << "\nplay" << std::endl;
  usbt.playCameraStreamCall(ctx);
  
  sleep_for(2s);
  
  std::cout << "\nstop" << std::endl;
  usbt.stopCameraStreamCall(ctx);
  
  std::cout << "\n thread stop" << std::endl;
  usbt.stopCall();
  
}


void test_4() {
    const char* name = "@TEST: usbthread_test: test 4: ";
    std::cout << name <<"** @@USBDeviceThread -> AVThread -> OpenGLThread **" << std::endl;
    // (USBDeviceThread:usbthread) --> {FrameFilter:info} --> {FifoFrameFilter:in_filter} -->> (AVThread:avthread) --> {InfoFrameFilter:decoded_info} -->> (OpenGLThread:glthread)

    OpenGLThread        glthread("gl_thread");
    FifoFrameFilter     &gl_in_filter = glthread.getFrameFilter();
    // InfoFrameFilter decoded_info("decoded",&gl_in_filter);
    DummyFrameFilter    decoded_info("decoded",false,&gl_in_filter); // non-verbose
    AVThread            avthread("avthread",decoded_info);
    FifoFrameFilter     &in_filter = avthread.getFrameFilter(); // request framefilter from AVThread
    // InfoFrameFilter out_filter("encoded",&in_filter);
    DummyFrameFilter    out_filter("encoded",false,&in_filter); // non-verbose
    USBDeviceThread     usbthread("usbthread");
    
    USBCameraConnectionContext ctx("/dev/video2", 1, &out_filter);
 
    std::cout << name << "starting threads" << std::endl;
    glthread.  startCall();
    avthread.  startCall();
    usbthread. startCall();
    
    avthread.  decodingOnCall();
    
    // create window
    Window window_id=glthread.createWindow();
    glthread.makeCurrent(window_id);
    std::cout << "new x window "<<window_id<<std::endl;

    // create render group & context
    glthread.newRenderGroupCall(window_id);
    int i=glthread.newRenderContextCall(1, window_id, 0);
    std::cout << "got render context id "<<i<<std::endl;
  
    std::cout << "\nplay" << std::endl;
    usbthread. playCameraStreamCall(ctx);
    
    sleep_for(60s);
    
    std::cout << "\nstop" << std::endl;
    usbthread.stopCameraStreamCall(ctx);
    
    sleep_for(5s);
    // sleep_for(604800s); //one week
    
    std::cout << name << "stopping threads" << std::endl;
    usbthread. stopCall();
    avthread.  stopCall();
    glthread.  stopCall();
}


void test_5() {
  const char* name = "@TEST: usbthread_test: test 5: ";
  std::cout << name <<"** @@USBDeviceThread -> AVThread -> OpenGLThread with 2 cameras **" << std::endl;
  
  // (USBDeviceThread:usbthread) --> {FrameFilter:info} --> {FifoFrameFilter:in_filter} -->> (AVThread:avthread) --> {InfoFrameFilter:decoded_info} -->> (OpenGLThread:glthread)

    USBDeviceThread     usbthread("usbthread");
    OpenGLThread        glthread("gl_thread");
    FifoFrameFilter     &gl_in_filter = glthread.getFrameFilter();
    // InfoFrameFilter decoded_info("decoded",&gl_in_filter);
    
    DummyFrameFilter    decoded_info("decoded",false,&gl_in_filter); // non-verbose
    AVThread            avthread("avthread",decoded_info);
    FifoFrameFilter     &in_filter = avthread.getFrameFilter(); // request framefilter from AVThread
    // InfoFrameFilter out_filter("encoded",&in_filter);
    DummyFrameFilter    out_filter("encoded",false,&in_filter); // non-verbose
    
    DummyFrameFilter    decoded_info2("decoded2",false,&gl_in_filter); // non-verbose
    AVThread            avthread2("avthread2",decoded_info2);
    FifoFrameFilter     &in_filter2 = avthread2.getFrameFilter(); // request framefilter from AVThread
    // InfoFrameFilter out_filter2("encoded2",&in_filter2);
    DummyFrameFilter    out_filter2("encoded2",false,&in_filter2); // non-verbose
    
    USBCameraConnectionContext ctx("/dev/video2", 1, &out_filter);
    USBCameraConnectionContext ctx2("/dev/video3", 2, &out_filter2);
    
    std::cout << name << "starting threads" << std::endl;
    glthread.  startCall();
    
    avthread.  startCall();
    avthread2. startCall();
    
    usbthread. startCall();

    avthread.  decodingOnCall();
    avthread2. decodingOnCall();
    
    // create windows
    Window window_id=glthread.createWindow();
    glthread.makeCurrent(window_id);
    std::cout << "new x window "<<window_id<<std::endl;

    Window window_id2=glthread.createWindow();
    glthread.makeCurrent(window_id2);
    std::cout << "new x window 2 "<<window_id2<<std::endl;
    
    // create render groups & contexts
    glthread.newRenderGroupCall(window_id);
    int i=glthread.newRenderContextCall(1, window_id, 0);
    std::cout << "got render context id "<<i<<std::endl;
  
    glthread.newRenderGroupCall(window_id2);
    int i2=glthread.newRenderContextCall(2, window_id2, 0);
    std::cout << "got render context id 2 "<<i2<<std::endl;
    
    std::cout << "\nplay" << std::endl;
    usbthread. playCameraStreamCall(ctx);
    usbthread. playCameraStreamCall(ctx2);
    
    sleep_for(60s);
    
    std::cout << "\nstop" << std::endl;
    usbthread.stopCameraStreamCall(ctx);
    usbthread.stopCameraStreamCall(ctx2);
    
    sleep_for(5s);
    // sleep_for(604800s); //one week
    
    std::cout << name << "stopping threads" << std::endl;
    usbthread. stopCall();
    avthread.  stopCall();
    glthread.  stopCall();
  
  
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


