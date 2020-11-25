/*
 * thread_test.cpp : Testing the Thread class using Test*Thread classes
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
 *  @file    thread_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.0.2 
 *  
 *  @brief Testing the Thread class using Test*Thread classes.  Compile with "make tests" and run with valgrind
 *
 */

#include "thread.h"
#include "framefilter.h"
#include "logging.h"
#include "avdep.h"
#include "test_import.h"


using namespace std::chrono_literals;
using std::this_thread::sleep_for;  // http://en.cppreference.com/w/cpp/thread/sleep_for

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() { // single producer, single consumer
    const char* name = "@TEST: threads_test: test 1: ";
    std::cout << name <<"** @@Consumer and producer **" << std::endl;
    
    bool start_consumer;
    FrameFifoContext ctx =FrameFifoContext();
    
    FrameFifo fifo("fifo",ctx);
    // start_consumer=false;
    
    start_consumer=true;
    
    TestProducerThread producer("producer",&fifo);
    TestConsumerThread consumer("consumer",&fifo);
    
    producer.startCall();
    
    sleep_for(3s); // enable this for first writing all frames and then reading them
    
    if (start_consumer) { consumer.startCall(); }
    
    sleep_for(3s);
    
    producer.stopCall();
    if (start_consumer) { consumer.stopCall(); }
    
}


void test_2() { // two producers, single consumer
    const char* name = "@TEST: threads_test: test 2: ";
    std::cout << name <<"** @@Consumer and producer 2 **" << std::endl;
    
    bool start_consumer, use2;
    FrameFifoContext ctx =FrameFifoContext();
    
    FrameFifo fifo("fifo",ctx);
    
    // use2=false;
    use2=true;
    
    // start_consumer=false;
    start_consumer=true;
    
    TestProducerThread producer ("producer1", &fifo, 1);
    TestProducerThread producer2("producer2", &fifo, 2);
    TestConsumerThread consumer ("consumer1", &fifo);
    
    producer.startCall();
    if (use2) {producer2.startCall();}
    
    sleep_for(3s); // enable this for first writing all frames and then reading them
    
    fifo.dumpFifo();
    
    if (start_consumer) { consumer.startCall(); }
    
    sleep_for(3s);
    
    producer.stopCall();
    if (use2) {producer2.stopCall();}
    if (start_consumer) { consumer.stopCall(); }
    
}


void test_3() {
}


void test_4() {
}


void test_5() {
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


