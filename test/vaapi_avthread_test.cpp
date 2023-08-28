/*
 * vaapi_avthread_test.cpp : Test classes VAAPIThread
 * 
 * Copyright 2017-2023 Sampsa Riikonen
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
 *  @file    vaapi_avthread_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2023
 *  @version 1.5.3 
 *  
 *  @brief Test classes VAAPIThread
 *
 */ 


// TODO:
// decoding from rtsp stream
// instantiate N vaapi decoders

#include "avthread.h"
#include "vaapithread.h"
#include "framefilter.h"
#include "logging.h"
#include "test_import.h"
#include "livethread.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for;  // http://en.cppreference.com/w/cpp/thread/sleep_for

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");

SetupFrame setupframe   =SetupFrame();   // video setup frame
SetupFrame setupframe_a =SetupFrame(); // audio setup frame

BasicFrame vframe     =BasicFrame(); // video frame
BasicFrame aframe     =BasicFrame(); // audio frame

void init_frames() {
    
    // VAAPIThread does not really care about the slot number
    // LiveThread is requested to send the frames with a certain slot number to a certain filter .. and that filter receives always frames with the same slot number
    // There should always be an VAAPIThread per slot
    
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
    // *****
    // filtergraph:
    // -->> [VAAPIThread:avthread] --> {FrameFilter:out_filter}
    // ****
    InfoFrameFilter out_filter("out");
    VAAPIThread avthread("avthread",out_filter);
    FifoFrameFilter &in_filter = avthread.getFrameFilter();

    const char* name = "@TEST: vaapi_avthread_test: test 1: ";
    std::cout << name <<"** @@Send a setup frame to VAAPIThread **" << std::endl;
    
    std::cout << name << "starting av thread" << std::endl;
    avthread.startCall();
    avthread.decodingOnCall();
    
    // in_fifo.dumpStack();
    
    sleep_for(2s);
    
    in_filter.run(&setupframe);
    
    sleep_for(5s);
    
    // avthread.getFifo().dumpStacks();
    
    std::cout << name << "stopping avthread" << std::endl;
    avthread.stopCall();
}


void test_2() {
    // filtergraph:
    // -->> [VAAPIThread:avthread] --> {FrameFilter:out_filter}
    // ****
    InfoFrameFilter out_filter("out");
    VAAPIThread avthread("avthread",out_filter);
    FifoFrameFilter &in_filter = avthread.getFrameFilter();

    const char* name = "@TEST: vaapi_avthread_test: test 2: ";
    std::cout << name <<"** @@Send a setup frame and two void video frames to VAAPIThread **" << std::endl;
    
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
    
    std::cout << name << "stopping avthread" << std::endl;
    avthread.stopCall();
}


void test_3() {
    // -->> [VAAPIThread:avthread] --> {FrameFilter:out_filter}
    InfoFrameFilter out_filter("out");
    VAAPIThread avthread("avthread",out_filter);
    FifoFrameFilter &in_filter = avthread.getFrameFilter();

    const char* name = "@TEST: vaapi_avthread_test: test 3: ";
    std::cout << name <<"** @@Send a void video frame to VAAPIThread (no setup frame) **" << std::endl;
    
    std::cout << name << "starting av thread" << std::endl;
    avthread.startCall();
    avthread.decodingOnCall();
    
    sleep_for(2s);
    
    in_filter.run(&vframe);
    in_filter.run(&vframe);
    
    sleep_for(5s);
    
    std::cout << name << "stopping avthread" << std::endl;
    avthread.stopCall();
}


void test_4() {
    // -->> [VAAPIThread:avthread] --> {FrameFilter:out_filter}
    InfoFrameFilter out_filter("out");
    VAAPIThread avthread("avthread",out_filter);
    FifoFrameFilter &in_filter = avthread.getFrameFilter();

    const char* name = "@TEST: vaapi_avthread_test: test 4: ";
    std::cout << name <<"** @@Send two consecutive setup frames to VAAPIThread (i.e. reinit) **" << std::endl;
    
    std::cout << name << "starting av thread" << std::endl;
    avthread.startCall();
    avthread.decodingOnCall();
    
    sleep_for(2s);
    
    in_filter.run(&setupframe);
    in_filter.run(&setupframe);
    in_filter.run(&vframe);
    in_filter.run(&vframe);
    
    sleep_for(5s);
    
    std::cout << name << "stopping avthread" << std::endl;
    avthread.stopCall();
}


void test_5() {
    // -->> [VAAPIThread:avthread] --> {FrameFilter:out_filter}
    InfoFrameFilter out_filter("out");
    VAAPIThread avthread("avthread",out_filter);
    FifoFrameFilter &in_filter = avthread.getFrameFilter();
    VAAPIThread avthread2("avthread2",out_filter);

    const char* name = "@TEST: vaapi_avthread_test: test 5: ";
    std::cout << name <<"** @@Start two VAAPIThreads **" << std::endl;
    
    std::cout << name << "starting av thread" << std::endl;
    
    avthread. startCall();
    avthread2.startCall();
    
    avthread. decodingOnCall();
    avthread2.decodingOnCall();
    
    sleep_for(2s);
    
    in_filter.run(&setupframe);
    
    sleep_for(5s);
    
    std::cout << name << "stopping avthread" << std::endl;
    
    avthread. stopCall();
    avthread2.stopCall();
}


void test_6() {
    // -->> [VAAPIThread:avthread] --> {FrameFilter:out_filter}
    InfoFrameFilter out_filter("out");
    VAAPIThread avthread("avthread",out_filter);
    FifoFrameFilter &in_filter = avthread.getFrameFilter();

    const char* name = "@TEST: vaapi_avthread_test: test 6: ";
    std::cout << name <<"** @@Send setup, video and audio frames to VAAPIThread **" << std::endl;
    
    std::cout << name << "starting av thread" << std::endl;
    avthread.startCall();
    avthread.decodingOnCall();
    
    sleep_for(2s);
    
    in_filter.run(&setupframe);
    in_filter.run(&setupframe_a);
    in_filter.run(&vframe);
    in_filter.run(&aframe);
    
    sleep_for(5s);
    
    std::cout << name << "stopping avthread" << std::endl;
    avthread.stopCall();
}

void test_7() {
    // (LiveThread:livethread) --> {FrameFilter:info} --> {FifoFrameFilter:in_filter} -->> (VAAPIThread:avthread) --> {SwScaleFrameFilter:sw_scale} --> {InfoFrameFilter:scaled}
    InfoFrameFilter out_filter("out");
    VAAPIThread avthread("avthread",out_filter);
    // AVThread avthread("avthread",out_filter);
    FifoFrameFilter &in_filter = avthread.getFrameFilter();
    LiveThread livethread("live");

    const char* name = "@TEST: vaapi_avthread_test: test 7: ";
    std::cout << name <<"** @@Full pipeline from rtsp connection to VAAPIThread **" << std::endl;

    if (!stream_1)
    {
        std::cout << name << "ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1" << std::endl;
        exit(2);
    }

    std::cout << name << "starting threads" << std::endl;
    livethread.startCall();
    avthread.startCall();

    avthread.decodingOnCall();

    //sleep_for(2s);
    std::cout << name << "registering stream" << std::endl;
    LiveConnectionContext ctx = LiveConnectionContext(LiveConnectionType::rtsp, std::string(stream_1), 2, &in_filter);
    livethread.registerStreamCall(ctx);

    // sleep_for(1s);
    std::cout << name << "playing stream !" << std::endl;
    livethread.playStreamCall(ctx);

    // sleep_for(2s);
    sleep_for(5s);
    // sleep_for(10s);
    // sleep_for(604800s); //one week

    std::cout << name << "stopping threads" << std::endl;
    livethread.stopCall();
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
            case(7):
                test_7();
                break;
            default:
                std::cout << "No such test "<<argcv[1]<<" for "<<argcv[0]<<std::endl;
        }
    }
} 

