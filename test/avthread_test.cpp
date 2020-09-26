/*
 * avthread_test.cpp : Test class AVThread
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
 *  @file    avthread_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.0.0 
 *  
 *  @brief Test class AVThread
 *
 */ 

#include "avthread.h"
#include "framefilter.h"
#include "logging.h"
#include "test_import.h"


using namespace std::chrono_literals;
using std::this_thread::sleep_for;  // http://en.cppreference.com/w/cpp/thread/sleep_for

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");

SetupFrame setupframe   =SetupFrame();   // video setup frame
SetupFrame setupframe_a =SetupFrame(); // audio setup frame

BasicFrame vframe     =BasicFrame(); // video frame
BasicFrame aframe     =BasicFrame(); // audio frame

// *****
// filtergraph:
// -->> [AVThread:avthread] --> {FrameFilter:out_filter}
// ****
InfoFrameFilter out_filter("out");
AVThread avthread("avthread",out_filter);
FifoFrameFilter &in_filter = avthread.getFrameFilter();

AVThread avthread2("avthread2",out_filter);


void init_frames() {
    
    // AVThread does not really care about the slot number
    // LiveThread is requested to send the frames with a certain slot number to a certain filter .. and that filter receives always frames with the same slot number
    // There should always be an AVThread per slot
    
    setupframe.sub_type=SetupFrameType::stream_init;
    setupframe.n_slot=1;
    setupframe.subsession_index =0;
    setupframe.media_type =AVMEDIA_TYPE_VIDEO; // what frame types are to be expected from this stream
    setupframe.codec_id   =AV_CODEC_ID_H264; // what frame types are to be expected from this stream
    
    vframe.n_slot=1;
    vframe.subsession_index =0;
    vframe.media_type =AVMEDIA_TYPE_VIDEO;
    vframe.codec_id   =AV_CODEC_ID_H264;
    vframe.payload.resize(10);
    vframe.payload={0,0,0,1,0,0,0,0,0,0};
    
    
    setupframe.sub_type=SetupFrameType::stream_init;
    setupframe_a.n_slot=1;
    setupframe_a.subsession_index =1;
    setupframe_a.media_type =AVMEDIA_TYPE_AUDIO; // what frame types are to be expected from this stream
    setupframe_a.codec_id   =AV_CODEC_ID_PCM_MULAW; // what frame types are to be expected from this stream
    
    aframe.n_slot=1;
    aframe.subsession_index =1;
    aframe.media_type =AVMEDIA_TYPE_AUDIO;
    aframe.codec_id   =AV_CODEC_ID_PCM_MULAW;
    aframe.payload.resize(10);
    aframe.payload={4,4,4,4,4,4,4,4,4,4};
}



void test_1() {
    const char* name = "@TEST: avthread_test: test 1: ";
    std::cout << name <<"** @@Send a setup frame to AVThread **" << std::endl;
    
    std::cout << name << "starting av thread" << std::endl;
    avthread.startCall();
    avthread.decodingOnCall();
    
    // in_fifo.dumpStack();
    
    sleep_for(2s);
    
    in_filter.run(&setupframe);
    
    sleep_for(5s);
    
    // avthread.getFifo().dumpStacks();
    
    std::cout << name << "stopping live thread" << std::endl;
    avthread.stopCall();
}


void test_2() {
    const char* name = "@TEST: avthread_test: test 2: ";
    std::cout << name <<"** @@Send a setup frame and two void video frames to AVThread **" << std::endl;
    
    std::cout << name << "starting av thread" << std::endl;
    avthread.startCall();
    avthread.decodingOnCall();
    
    sleep_for(2s);
    
    std::cout << name << "\nWriting setup frame\n";
    in_filter.run(&setupframe);
    std::cout << name << "\nWriting H264 frame\n";
    in_filter.run(&vframe);
    std::cout << name << "\nWriting H264 frame\n";
    in_filter.run(&vframe);
    
    sleep_for(5s);
    
    std::cout << name << "stopping live thread" << std::endl;
    avthread.stopCall();
}


void test_3() {
    const char* name = "@TEST: avthread_test: test 3: ";
    std::cout << name <<"** @@Send a void video frame to AVThread (no setup frame) **" << std::endl;
    
    std::cout << name << "starting av thread" << std::endl;
    avthread.startCall();
    avthread.decodingOnCall();
    
    sleep_for(2s);
    
    in_filter.run(&vframe);
    in_filter.run(&vframe);
    
    sleep_for(5s);
    
    std::cout << name << "stopping live thread" << std::endl;
    avthread.stopCall();
}


void test_4() {
    const char* name = "@TEST: avthread_test: test 4: ";
    std::cout << name <<"** @@Send two consecutive setup frames to AVThread (i.e. reinit) **" << std::endl;
    
    std::cout << name << "starting av thread" << std::endl;
    avthread.startCall();
    avthread.decodingOnCall();
    
    sleep_for(2s);
    
    in_filter.run(&setupframe);
    in_filter.run(&setupframe);
    in_filter.run(&vframe);
    in_filter.run(&vframe);
    
    sleep_for(5s);
    
    std::cout << name << "stopping live thread" << std::endl;
    avthread.stopCall();
}


void test_5() {
    const char* name = "@TEST: avthread_test: test 5: ";
    std::cout << name <<"** @@Start two AVThreads **" << std::endl;
    
    std::cout << name << "starting av thread" << std::endl;
    
    avthread. startCall();
    avthread2.startCall();
    
    avthread. decodingOnCall();
    avthread2.decodingOnCall();
    
    sleep_for(2s);
    
    in_filter.run(&setupframe);
    
    sleep_for(5s);
    
    std::cout << name << "stopping live thread" << std::endl;
    
    avthread. stopCall();
    avthread2.stopCall();
}


void test_6() {
    const char* name = "@TEST: avthread_test: test 6: ";
    std::cout << name <<"** @@Send setup, video and audio frames to AVThread **" << std::endl;
    
    std::cout << name << "starting av thread" << std::endl;
    avthread.startCall();
    avthread.decodingOnCall();
    
    sleep_for(2s);
    
    in_filter.run(&setupframe);
    in_filter.run(&setupframe_a);
    in_filter.run(&vframe);
    in_filter.run(&aframe);
    
    sleep_for(5s);
    
    std::cout << name << "stopping live thread" << std::endl;
    avthread.stopCall();
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
        
        init_frames();
        
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

