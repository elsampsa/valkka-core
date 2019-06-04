/*
 * cache_test.cpp : Test caching frames
 * 
 * Copyright 2018 Valkka Security Ltd. and Sampsa Riikonen.
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
 *  @file    cache_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2018
 *  @version 0.11.0 
 *  
 *  @brief   Test caching frames
 *
 */ 

#include "framefifo.h"
#include "framefilter.h"
#include "cachestream.h"
#include "logging.h"
#include "avdep.h"
#include "test_import.h"


using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
  
    const char* name = "@TEST: cache_test: test 1: ";
    std::cout << name <<"** @@Instantiate FrameCache, insert frames **" << std::endl;

    FrameCacheContext ctx = FrameCacheContext();

    FrameCache cache = FrameCache("test", ctx);

    std::vector<BasicFrame*> frames;
    int i;
    for(i=0; i<100; i++) {
        std::cout << i << std::endl;
        BasicFrame *f = new BasicFrame();
        frames.push_back(f);
        
        f->payload.resize(100, i);
        f->mstimestamp          = 5000+i;
        f->media_type           =AVMEDIA_TYPE_VIDEO;
        f->codec_id             =AV_CODEC_ID_H264;
        f->subsession_index     =0;
        
        H264Pars p = H264Pars();
        
        if (i%5 == 0) {
            p.slice_type = H264SliceType::sps;
        }
        else {
            p.slice_type = H264SliceType::i;
        }
        
        f->h264_pars = p;
        
        cache.writeCopy(f);
    }

    Frame* f;
    
    std::cout << "isEmpty = " << cache.isEmpty() << std::endl;
    cache.dump();

    std::cout << "seek" << std::endl;
    std::cout << cache.seek(5013) << std::endl;
    f = cache.pullNextFrame();
    std::cout << *f << std::endl;
    
    std::cout << "seek" << std::endl;
    cache.seek(5015);
    f = cache.pullNextFrame();
    std::cout << *f << std::endl;
    
    std::cout << "seek" << std::endl;
    cache.seek(5016);
    f = cache.pullNextFrame();
    std::cout << *f << std::endl;
    
    std::cout << "pullNextFrame" << std::endl;
    f = cache.pullNextFrame();
    std::cout << *f << std::endl;
    f = cache.pullNextFrame();
    std::cout << *f << std::endl;
    f = cache.pullNextFrame();
    std::cout << *f << std::endl;
    
    std::cout << "seek and pull over the edge" << std::endl;
    cache.seek(5098);
    f = cache.pullNextFrame();
    std::cout << *f << std::endl;
    f = cache.pullNextFrame();
    std::cout << (long int)f << std::endl;
    f = cache.pullNextFrame();
    std::cout << (long int)f << std::endl;
    f = cache.pullNextFrame();
    std::cout << (long int)f << std::endl;
    f = cache.pullNextFrame();
    std::cout << (long int)f << std::endl;
    f = cache.pullNextFrame();
    std::cout << (long int)f << std::endl;
    f = cache.pullNextFrame();
    std::cout << (long int)f << std::endl;
    f = cache.pullNextFrame();
    std::cout << (long int)f << std::endl;
    
    std::cout << "seek over the edge" << std::endl;
    std::cout << cache.seek(6000) << std::endl;
 
    std::cout << "seek under the edge" << std::endl;
    std::cout << cache.seek(1000) << std::endl;
    
    for(auto it=frames.begin(); it!=frames.end(); it++) {
        delete *it;
    }
}


void test_2() {
    const char* name = "@TEST: cache_test: test 2: ";
    std::cout << name <<"** @@Test FrameCache **" << std::endl;
    
    FrameCache framecache = FrameCache("test", FrameCacheContext());
    CacheFrameFilter input("test", &framecache);

    std::vector<BasicFrame*> frames;
    int i;
    for(i=0; i<100; i++) {
        std::cout << i << std::endl;
        BasicFrame *f = new BasicFrame();
        frames.push_back(f);
        
        f->payload.resize(100, i);
        f->mstimestamp          = 5000+i;
        f->media_type           =AVMEDIA_TYPE_VIDEO;
        f->codec_id             =AV_CODEC_ID_H264;
        f->subsession_index     =0;
        
        H264Pars p = H264Pars();
        
        if (i%5 == 0) {
            p.slice_type = H264SliceType::sps;
        }
        else {
            p.slice_type = H264SliceType::i;
        }
        
        f->h264_pars = p;
        
        input.run(f);
    }
    
    framecache.dump();
    std::cout << "min, max time = " << framecache.getMinTime_() << " " << framecache.getMaxTime_() << std::endl;
    
    framecache.seek(5015);
    // TODO: test update, etc
    
    for(auto it=frames.begin(); it!=frames.end(); it++) {
        delete *it;
    }
}


void test_3() {
  
  const char* name = "@TEST: cache_test: test 3: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}


void test_4() {
  
  const char* name = "@TEST: cache_test: test 4: ";
  std::cout << name <<"** @@DESCRIPTION **" << std::endl;
  
}


void test_5() {
  
  const char* name = "@TEST: cache_test: test 5: ";
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


