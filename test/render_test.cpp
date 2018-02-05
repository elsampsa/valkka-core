/*
 * render_test.cpp : Test OpenGLThread rendering routines
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
 *  @file    render_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.0 
 *  
 *  @brief   Test OpenGLThread rendering routines
 *
 */ 

#include "opengl.h"
#include "openglthread.h"
#include "filters.h"
#include "logging.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for;

const char* stream_1   =std::getenv("VALKKA_TEST_RTSP_1");
const char* stream_2   =std::getenv("VALKKA_TEST_RTSP_2");
const char* stream_sdp =std::getenv("VALKKA_TEST_SDP");


void test_1() {
  const char* name = "@TEST: render_test: test 1: ";
  std::cout << name <<"** @@Loading YUV to PBO **" << std::endl;
  
  Frame* frame;
  YUVPBO* pbo;
  int size, ysize;
  GLubyte *y, *u, *v;
  
  Window window_id, window_id2, window_id3;
  Display* display;
  int idn;
  
  OpenGLThread glthread("glthread",/*n720p*/10,/*n1080p*/0,/*n1440p*/0,/*4K*/0,/*naudio*/0,/*msbuftime*/0,/*core_id*/-1);
  
  size=readyuvbytes("../aux/1.yuv",y,u,v);
  ysize=(size*2)/3;
  
  glthread.preRun();
  
  // pbo=glthread.Stack720p.get();
  // pbo=glthread.infifo.stack_720p[0];
  // frame=glthread.getFrame720p();
  frame=glthread.getFrame(BitmapPars::N720::type);
  pbo=frame->yuvpbo;
  loadYUVPBO(pbo, ysize, y, u, v); // load payload into PBO
  
  glthread.postRun();
}


void test_2() {
  const char* name = "@TEST: render_test: test 2: ";
  std::cout << name <<"** @@Create a RenderGroup and RenderContext **" << std::endl;
  Window window_id;
  int idn;
  
  OpenGLThread glthread("glthread",/*n720p*/10,/*n1080p*/0,/*n1440p*/0,/*4K*/0,/*naudio*/0,/*msbuftime*/0,/*core_id*/-1);
  glthread.preRun();
  
  window_id=glthread.createWindow();
  glthread.makeCurrent(window_id);
  
  std::cout << "new x window "<<window_id<<std::endl;
  
  // display         =glthread.getDisplayId();
  // const GLXContext& glc =glthread.getGlc(); 
  
  glthread.newRenderGroup(window_id);  
  
  idn=glthread.newRenderContext(1,window_id,0); // slot 1 goes to x window window_id
  std::cout << "new Render Context: " << idn << std::endl;
  
  YUVFramePars yuv_pars={BitmapPars::N720::type,AV_PIX_FMT_YUV420P,BitmapPars::N720::w,BitmapPars::N720::h};
  
  glthread.activateSlot(1,yuv_pars); // i.e. when config frame arrives .. vector slots_[n_slot] initialized
  
  glthread.reportSlots(); // show active slots, slots with RenderGroups and their RenderContexes
  glthread.reportRenderGroups();
  glthread.reportRenderList();
}


void test_3() {
  const char* name = "@TEST: render_test: test 3: ";
  std::cout << name <<"** @@Create and delete RenderGroup(s) and RenderContex(es) **" << std::endl;
  Window window_id, window_id2, window_id3;
  int idn, idn2, idn3;
  
  OpenGLThread glthread("glthread",/*n720p*/10,/*n1080p*/0,/*n1440p*/0,/*4K*/0,/*naudio*/0,/*msbuftime*/0,/*core_id*/-1);
  
  glthread.preRun();
  
  window_id=glthread.createWindow();
  glthread.makeCurrent(window_id);
  
  window_id2=glthread.createWindow();
  glthread.makeCurrent(window_id2);
  
  window_id3=glthread.createWindow();
  glthread.makeCurrent(window_id3);  
  
  std::cout << "new x window   "<<window_id<<std::endl;
  std::cout << "new x window 2 "<<window_id2<<std::endl;
  std::cout << "new x window 3 "<<window_id3<<std::endl;
  
  glthread.newRenderGroup(window_id);
  glthread.newRenderGroup(window_id3);
  
  /*
  std::cout << "has group?     "<<window_id<<" "<<glthread.hasRenderGroup(window_id) <<std::endl;
  std::cout << "has group?     "<<window_id<<" "<<glthread.hasRenderGroup(window_id) <<std::endl;
  std::cout << "deleted group? "<<window_id<<" "<<glthread.delRenderGroup(window_id) <<std::endl;
  std::cout << "has group?     "<<window_id<<" "<<glthread.hasRenderGroup(window_id) <<std::endl;
  return;
  */
  
  idn=glthread.newRenderContext(1,window_id,0); // slot, x window, z
  std::cout << "new Render Context: " << idn << std::endl;
  
  idn2=glthread.newRenderContext(2,window_id2,0); // should be discarded (noo group window_id2)
  std::cout << "new Render Context: " << idn << std::endl;
  
  idn3=glthread.newRenderContext(2,window_id3,0);
  std::cout << "new Render Context: " << idn << std::endl;
  
  YUVFramePars yuv_pars={BitmapPars::N720::type,AV_PIX_FMT_YUV420P,BitmapPars::N720::w,BitmapPars::N720::h};
  
  glthread.activateSlot(1,yuv_pars);
  
  glthread.reportSlots(); // show active slots, slots with RenderGroups and their RenderContexes
  glthread.reportRenderGroups();
  glthread.reportRenderList();
  
  glthread.delRenderContext(idn);
  std::cout << "deleted RenderContext " << idn <<std::endl<<std::endl;
  
  glthread.reportSlots(); // show active slots, slots with RenderGroups and their RenderContexes
  glthread.reportRenderGroups();
  glthread.reportRenderList();
  
  bool dl=false;
  dl=glthread.delRenderGroup(window_id);
  std::cout << "deleted RenderGroup " << window_id << " " << dl <<std::endl<<std::endl;
  
  glthread.reportSlots(); // show active slots, slots with RenderGroups and their RenderContexes
  glthread.reportRenderGroups();
  glthread.reportRenderList();
  
  dl=false;
  dl=glthread.delRenderGroup(window_id);
  std::cout << "deleted RenderGroup " << window_id << " " << dl <<std::endl<<std::endl;
  
  dl=false;
  dl=glthread.delRenderGroup(window_id3);
  std::cout << "deleted RenderGroup " << window_id3 << " " << dl <<std::endl<<std::endl;
  
  glthread.reportSlots(); // show active slots, slots with RenderGroups and their RenderContexes
  glthread.reportRenderGroups();
  glthread.reportRenderList();
  
}


void test_4() {
  const char* name = "@TEST: render_test: test 4: ";
  std::cout << name <<"** @@Create RenderGroup, RenderContext and render **" << std::endl;
  Window window_id;
  int idn;
  Frame* frame;
  YUVPBO* pbo;
  int size, ysize;
  GLubyte *y, *u, *v;
  
  OpenGLThread glthread("glthread",/*n720p*/10,/*n1080p*/0,/*n1440p*/0,/*4K*/0,/*naudio*/0,/*msbuftime*/0,/*core_id*/-1);
  glthread.preRun();
  
  glthread.reportStacks();
  
  // get a pbo instance
  // pbo=glthread.Stack720p.get();
  // pbo=glthread.infifo.stack_720p[0];
  // frame=glthread.getFrame720p();
  frame=glthread.getFrame(BitmapPars::N720::type);
  std::cout << "Got frame  " << *frame;
  pbo=frame->yuvpbo;
  std::cout << "Got yuvpbo " << *pbo;
  
  // return;
  
  // load YUV
  size=readyuvbytes("../aux/1.yuv",y,u,v);
  ysize=(size*2)/3;
  
  std::cout << "File size "<< size <<" pbo size "<< pbo->size << std::endl;
  
  // zeroyuvbytes(ysize,y,u,v); // so.. zero data creates a green screen
  
  loadYUVPBO(pbo, ysize, y, u, v); // load payload into PBO
  // return;
  peekYUVPBO(pbo);
  // return;
  
  // create x window
  window_id=glthread.createWindow();
  glthread.makeCurrent(window_id);
  std::cout << "new x window "<<window_id<<std::endl;
  
  // create render group and context
  glthread.newRenderGroup(window_id);  
  idn=glthread.newRenderContext(1,window_id,0); // slot 1 goes to x window window_id
  std::cout << "new Render Context: " << idn << std::endl;
  
  // check that everythings in place
  glthread.reportSlots(); // show active slots, slots with RenderGroups and their RenderContexes
  glthread.reportRenderGroups();
  glthread.reportRenderList();
 
  YUVFramePars yuv_pars={BitmapPars::N720::type,AV_PIX_FMT_YUV420P,BitmapPars::N720::w,BitmapPars::N720::h};
  glthread.activateSlot(1,yuv_pars);
  
  // frame => slot 1
  // slots_[1] => if active, slots_[1]->loadTEX(frames PBO instance here)
  // TEX loaded, time to render
  // render_groups[1]
  
  glthread.loadTEX(1,pbo);
  
  int i;
  for(i=0; i<100; i++) {
    glthread.render(1);
    sleep_for(0.1s);
  }
}


void test_5() {
  
  const char* name = "@TEST: render_test: test 5: ";
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




