/*
 * openglthread.cpp : FrameFifo for OpenGLThread: stack of YUV frames and uploading to GPU
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
 *  @file    openglthread.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.17.5 
 *  
 *  @brief FrameFifo for OpenGLThread: stack of YUV frames and uploading to GPU
 *
 *  @section DESCRIPTION
 *  
 */ 

#include "openglframefifo.h"
#include "logging.h"
#include "tools.h"


OpenGLFrameFifo::OpenGLFrameFifo(OpenGLFrameFifoContext gl_ctx) : gl_ctx(gl_ctx), FrameFifo("opengl",FrameFifoContext(0, 0, 0, 0, 20, 20, gl_ctx.flush_when_full)),  debug(false) {

  // from FrameFifo:
  //
  // std::map<FrameClass,Reservoir>  reservoirs
  // std::map<FrameClass,Stack>      stacks
  //
  // => reservoirs, stacks
  //
  // here we define additionally:
  //
  // std::map<BitmapType,Reservoir>  yuv_reservoirs;
  // std::map<BitmapType,Stack>      yuv_stacks;
  //
  // => yuv_reservoirs, yuv_stacks
  //
  
  // some "dummy" variables
  YUVReservoir reservoir; // std::maps will make it's own copy of this
  YUVStack     stack;     // std::maps will make it's own copy of this
  
#define make_reservoir_and_stack_yuv(PARNAME) {\
  yuv_reservoirs.insert(std::make_pair(PARNAME.type, reservoir));\
  yuv_stacks.insert(std::make_pair(PARNAME.type, stack));\
}

  make_reservoir_and_stack_yuv(N720);
  make_reservoir_and_stack_yuv(N1080);
  make_reservoir_and_stack_yuv(N1440);
  make_reservoir_and_stack_yuv(N4K);
  
  // Mother class FrameFifo has reservoirs[FrameClass:setup], reservoir[FrameClass::signal]
}


OpenGLFrameFifo::~OpenGLFrameFifo() {
  opengllogger.log(LogLevel::debug) << "OpenGLFrameFifo: destructor "<<std::endl;
}


void OpenGLFrameFifo::allocateYUV() {
  int i;
  
#define allocate_reservoir_and_stack_yuv(PARNAME, NUM) {\
  for(i=0;i<NUM;i++) {\
    yuv_reservoirs[PARNAME.type].push_back(new YUVFrame(PARNAME));\
    yuv_stacks[PARNAME.type].push_back(yuv_reservoirs[PARNAME.type].back());\
  }\
}

  // Mother class FrameFifo has reservoirs[FrameClass:setup], reservoir[FrameClass::signal]
  // allocate_reservoir_and_stack_yuv(N1080, gl_ctx.n_1080p); // test: reverse allocation order
  allocate_reservoir_and_stack_yuv(N720,  gl_ctx.n_720p);
  allocate_reservoir_and_stack_yuv(N1080, gl_ctx.n_1080p);
  allocate_reservoir_and_stack_yuv(N1440, gl_ctx.n_1440p);
  allocate_reservoir_and_stack_yuv(N4K,   gl_ctx.n_4K);
  
  // YUVFrame::YUVPBO instances will be reserved at OpenGLThread::preRun (i.e. after the OpenGLThread has started)
#ifdef VALGRIND_GPU_DEBUG
#else
  glFinish();
#endif
}
  

void OpenGLFrameFifo::deallocateYUV() {
  // recycleAll(); // nopes .. this is for BasicFrame, etc.
  
#define deallocate_reservoir_yuv(PARNAME) {\
  YUVReservoir &reservoir=yuv_reservoirs[PARNAME.type];\
  for (auto it=reservoir.begin(); it!=reservoir.end(); ++it) { delete *(it); };\
}
  
  deallocate_reservoir_yuv(N720);
  deallocate_reservoir_yuv(N1080);
  deallocate_reservoir_yuv(N1440);
  deallocate_reservoir_yuv(N4K);
  
#ifdef VALGRIND_GPU_DEBUG
#else
  glFinish();
#endif
}


YUVFrame* OpenGLFrameFifo::prepareAVBitmapFrame(AVBitmapFrame* bmframe) {// Go from AVBitmapFrame to YUVFrame
  BitmapType bmtype;
  const BitmapPars &bmpars =bmframe->bmpars;
  
  // dumpYUVStacks();
  
  if (bmpars.y_size <= N720.y_size)  { // frames obtained with getFrame will be recycled by the presentation routine
    bmtype =N720.type;
  }
  else if (bmpars.y_size <= N1080.y_size) {
    bmtype =N1080.type;
  }
  else if (bmpars.y_size <= N1440.y_size) {
    bmtype =N1440.type;
  }
  else if (bmpars.y_size <= N4K.y_size)   {
    bmtype =N4K.type;
  }
  else {
    opengllogger.log(LogLevel::fatal) << "OpenGLFrameFifo: prepareAVFrame:  WARNING! Could not get large enough frame: "<< *bmframe <<std::endl;
    return NULL;
  }
  
  YUVStack &stack = yuv_stacks.at(bmtype);
  YUVFrame *yuvframe =NULL;
  
  { // mutex protected 
    std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
    if (stack.empty()) {
      opengllogger.log(LogLevel::normal) << "OpenGLFrameFifo: prepareAVFrame: OVERFLOW! No more frames in stack for bitmap type "<< bmtype <<std::endl;
      return NULL;
    }
    yuvframe =stack.front(); 
    stack.pop_front(); // .. remove that pointer from the stack
  } // mutex protected
  yuvframe->fromAVBitmapFrame(bmframe);
  
  return yuvframe;
}


bool OpenGLFrameFifo::writeCopy(Frame* f, bool wait) {
  // Decoder gives AVBitmapFrame => OpenGLThread internal FrameFilter => OpenGLThread FrameFifo::writeCopy(Frame *f) => .. returns once a copy has been made
  if (f->getFrameClass()!=FrameClass::avbitmap) {
    return FrameFifo::writeCopy(f,wait); // call motherclass "standard" writeCopy
  }

  YUVFrame *yuvframe =prepareAVBitmapFrame(static_cast<AVBitmapFrame*>(f)); // uploads AVBitmapFrame to GPU and returns 
    
  if (!yuvframe) {
    // no frame was taken from the YUVStack's.  No need to call recycle
    opengllogger.log(LogLevel::debug) << "OpenGLFrameFifo: writeCopy: WARNING! could not stage frame "<< *f <<std::endl;
    // if (opengllogger.log_level>=LogLevel::debug) { dumpYUVStacks(); }
    if (opengllogger.log_level>=LogLevel::debug) { YUVdiagnosis(); }
    return false;
  }
  
#ifdef TIMING_VERBOSE
  long int dt;
  dt=(getCurrentMsTimestamp()-yuvframe->mstimestamp);
  if (dt>100) {
    std::cout << "OpenGLFrameFifo: writeCopy : timing : inserting frame " << dt << " ms late" << std::endl;
  }
#endif
  
  if (debug) {
    opengllogger.log(LogLevel::normal) << "OpenGLFrameFifo: writeCopy: DEBUG MODE: recycling frame "<< *yuvframe <<std::endl;
    recycle(yuvframe);
    // reportStacks();
    return true;
  }
  
  { // mutex protection
    std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
    fifo.push_front(yuvframe);
    this->condition.notify_one(); // after receiving 
    return true;
  } // mutex protection
  
}


void OpenGLFrameFifo::recycle_(Frame* f) {// Return Frame f back into the stack.
  
  if (f->getFrameClass()!=FrameClass::yuv) {
    FrameFifo::recycle_(f); // call motherclass "standard" recycle
    return;
  }
    
  YUVFrame *yuvframe =static_cast<YUVFrame*>(f);
  
  YUVStack &yuv_stack = yuv_stacks.at(yuvframe->bmpars.type);
  yuv_stack.push_back(yuvframe); // take: from the front.  recycle: to the back
}


void OpenGLFrameFifo::dumpYUVStacks() {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  YUVStack stack;
  
  std::cout << "OpenGLFrameFifo : dumpYUVStacks : " << std::endl;
  for(auto it=yuv_stacks.begin(); it!=yuv_stacks.end(); ++it) {
    std::cout << "OpenGLFrameFifo : dumpYUVStacks : YUVStack=" << int(it->first) << std::endl;
    stack=it->second;
    for (auto its=stack.begin(); its!=stack.end(); ++its) {
      std::cout << "OpenGLFrameFifo : dumpYUVStacks :  " << *(*its) << std::endl;
    }
  }
  std::cout << "OpenGLFrameFifo : dumpYUVStacks : " << std::endl;
}
  

void OpenGLFrameFifo::YUVdiagnosis() {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  YUVStack stack;
  
  std::cout << "FrameFifo : YUVdiagnosis : " << std::endl;
  std::cout << "FrameFifo : YUVdiagnosis : Fifo  : " << fifo.size() << std::endl;
  std::cout << "FrameFifo : YUVdiagnosis : YUVStack : ";
  for(auto it=yuv_stacks.begin(); it!=yuv_stacks.end(); ++it) {
    std::cout << int(it->first) << ":" << (it->second).size() << ", ";
  }
  std::cout << std::endl;
  std::cout << "FrameFifo : YUVdiagnosis : " << std::endl;
}


  
