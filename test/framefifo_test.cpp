/*
 * framefifo_test.cpp : Testing the fifo classes
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
 *  @file    framefifo_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.5.2 
 *  
 *  @brief Testing fifo classes.  Compile with "make tests" and run with valgrind
 *
 */

#include "framefifo.h"
#include "framefilter.h"
#include "logging.h"
#include "avdep.h"
#include "test_import.h"


const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
  const char* name = "@TEST: fifo_test: test 1: ";
  std::cout << name <<"** @@Basic fifo tests **" << std::endl;
  
  FrameFifoContext ctx = FrameFifoContext();
  ctx.n_basic=10;
  ctx.flush_when_full=False;
  
  FrameFifo fifo("fifo",ctx);
  BasicFrame *f = new BasicFrame();
  
  f->resize(10);
  
  std::cout << "insert" << std::endl;
  fifo.writeCopy(f);
  fifo.dumpStacks(); 
  fifo.dumpFifo();
  std::cout << std::endl;
  
  f->resize(20);
  
  std::cout << "insert" << std::endl;
  fifo.writeCopy(f);
  fifo.dumpStacks(); 
  fifo.dumpFifo();
  std::cout << std::endl;
  
  f->resize(30);
  
  std::cout << "insert" << std::endl;
  fifo.writeCopy(f);
  fifo.dumpStacks(); 
  fifo.dumpFifo();
  std::cout << std::endl;
  
  Frame *ff;
  
  std::cout << "read" << std::endl;
  ff=fifo.read();
  fifo.dumpStacks(); 
  fifo.dumpFifo();
  std::cout << std::endl;
  // do stuff with the frame
  
  std::cout << "recycle" << std::endl;
  fifo.recycle(ff);
  fifo.dumpStacks(); 
  fifo.dumpFifo();
  std::cout << std::endl;
  
  delete f;
}


void test_2() {
  const char* name = "@TEST: fifo_test: test 2: ";
  std::cout << name <<"** @@Fifo overflow (no flush)**" << std::endl;
  
  FrameFifoContext ctx = FrameFifoContext();
  ctx.n_basic=10;
  ctx.flush_when_full=False;
  
  FrameFifo fifo("fifo",ctx);
  BasicFrame *f = new BasicFrame();
  
  int i;
  for(i=1; i<=20; i++) {
    f->resize(10*i);
    std::cout << "insert" << std::endl;
    fifo.writeCopy(f);
    fifo.dumpStacks(); 
    fifo.dumpFifo();
    std::cout << std::endl;
  }

  delete f;
}


void test_3() {
  const char* name = "@TEST: fifo_test: test 3: ";
  std::cout << name <<"** @@Fifo overflow (with flush)**" << std::endl;
  
  FrameFifoContext ctx = FrameFifoContext();
  ctx.n_basic=10;
  ctx.flush_when_full=True;
  
  FrameFifo fifo("fifo",ctx);
  BasicFrame *f = new BasicFrame();
  
  int i;
  for(i=1; i<=20; i++) {
    f->resize(10*i);
    std::cout << "insert" << std::endl;
    fifo.writeCopy(f);
    fifo.dumpStacks(); 
    fifo.dumpFifo();
    std::cout << std::endl;
  }
  
  delete f;
}


void test_4() {
  const char* name = "@TEST: fifo_test: test 4: ";
  std::cout << name <<"** @@Fifo overflow (with flush) : diagnosis output **" << std::endl;
  
  FrameFifoContext ctx = FrameFifoContext();
  ctx.n_basic=10;
  ctx.flush_when_full=True;
  
  FrameFifo fifo("fifo",ctx);
  BasicFrame *f = new BasicFrame();
  
  int i;
  for(i=1; i<=20; i++) {
    f->resize(10*i);
    std::cout << "insert" << std::endl;
    fifo.writeCopy(f);
    fifo.diagnosis();
    std::cout << std::endl;
  }
  
  delete f;
  
}


void test_5() {
  const char* name = "@TEST: fifo_test: test 5: ";
  std::cout << name <<"** @@Basic fifo tests: with different frame classes **" << std::endl;
  
  FrameFifoContext ctx = FrameFifoContext();
  ctx.n_basic=5;
  ctx.n_setup=5;
  ctx.flush_when_full=False;
  
  FrameFifo fifo("fifo",ctx);
  BasicFrame *bf = new BasicFrame();
  SetupFrame *sf = new SetupFrame();
  
  bf->resize(10);
  
  std::cout << "\n** INSERTING BASICFRAMES **\n" << std::endl;
  std::cout << "insert" << std::endl;
  fifo.writeCopy(bf);
  fifo.dumpStacks(); 
  fifo.dumpFifo();
  fifo.diagnosis();
  std::cout << std::endl;
  
  bf->resize(20);
  
  std::cout << "insert" << std::endl;
  fifo.writeCopy(bf);
  fifo.dumpStacks(); 
  fifo.dumpFifo();
  fifo.diagnosis();
  std::cout << std::endl;
  
  bf->resize(30);
  
  std::cout << "insert" << std::endl;
  fifo.writeCopy(bf);
  fifo.dumpStacks(); 
  fifo.dumpFifo();
  fifo.diagnosis();
  std::cout << std::endl;
  
  
  std::cout << "\n** INSERTING SETUPFRAMES **\n" << std::endl;
  std::cout << "insert" << std::endl;
  fifo.writeCopy(sf);
  fifo.dumpStacks(); 
  fifo.dumpFifo();
  fifo.diagnosis();
  std::cout << std::endl;
  
  std::cout << "insert" << std::endl;
  fifo.writeCopy(sf);
  fifo.dumpStacks(); 
  fifo.dumpFifo();
  fifo.diagnosis();
  std::cout << std::endl;
  
  std::cout << "insert" << std::endl;
  fifo.writeCopy(sf);
  fifo.dumpStacks(); 
  fifo.dumpFifo();
  fifo.diagnosis();
  std::cout << std::endl;
  
  
  Frame *ff;
  
  std::cout << "read" << std::endl;
  ff=fifo.read();
  fifo.dumpStacks(); 
  fifo.dumpFifo();
  fifo.diagnosis();
  std::cout << std::endl;
  // do stuff with the frame
  
  std::cout << "recycle" << std::endl;
  fifo.recycle(ff);
  fifo.dumpStacks(); 
  fifo.dumpFifo();
  fifo.diagnosis();
  std::cout << std::endl;
  
  delete bf;
  delete sf;
  
  
}


void test_6() {
    const char* name = "@TEST: fifo_test: test 6: ";
    std::cout << name <<"** @@Test FileDescriptor FrameFifo (FDFrameFifo) **" << std::endl;

    FrameFifoContext ctx = FrameFifoContext();
    ctx.n_basic = 10;
    ctx.flush_when_full = False;

    FDFrameFifo fifo("fifo", ctx);
    BasicFrame *f = new BasicFrame();

    f->resize(10);

    
    std::cout << "insert 1" << std::endl;
    fifo.writeCopy(f);
    //fifo.dumpStacks(); 
    //fifo.dumpFifo();
    std::cout << std::endl;

    f->resize(20);

    std::cout << "insert 2" << std::endl;
    fifo.writeCopy(f);
    //fifo.dumpStacks(); 
    //fifo.dumpFifo();
    std::cout << std::endl;

    f->resize(30);

    std::cout << "insert 3" << std::endl;
    fifo.writeCopy(f);
    //fifo.dumpStacks(); 
    //fifo.dumpFifo();
    std::cout << std::endl;

    Frame *ff;

    int fd = fifo.getFD();
    ssize_t i;
    uint64_t out;
    
    int cc;
    
    for(cc=0; cc<4; cc++) {
        std::cout << "read " << cc+1 << std::endl;
        i = read(fd, &out, sizeof(uint64_t));
        std::cout << "got: " << i << std::endl;
        if (i > 0) {
            ff = fifo.read();
            std::cout << "frame: " << *ff << std::endl;
            //fifo.dumpStacks(); 
            //fifo.dumpFifo();
            std::cout << std::endl;
            // do stuff with the frame

            std::cout << "recycle" << std::endl;
            fifo.recycle(ff);
            //fifo.dumpStacks(); 
            //fifo.dumpFifo();
            std::cout << std::endl;
        }
    }
    
    delete f;
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
      default:
        std::cout << "No such test "<<argcv[1]<<" for "<<argcv[0]<<std::endl;
    }
  }
} 


