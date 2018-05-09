/*
 * framefifo.cpp :
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
 *  @file    framefifo.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.4.0 
 *  
 *  @brief 
 */ 

#include "framefifo.h"
#include "enumiter.h"


FrameFifo::FrameFifo(const char *name, FrameFifoContext ctx) : name(name), ctx(ctx) {
  int i;
  
  // some "dummy" variables
  Reservoir reservoir; // std::maps will make it's own copy of this
  Stack     stack;     // std::maps will make it's own copy of this
  
#define make_reservoir_and_stack(NAME, PARNAME, CLASSNAME) {\
  reservoirs.insert(std::make_pair(FrameClass::NAME, reservoir));\
  stacks.insert(std::make_pair(FrameClass::NAME, stack));\
  for(i=0;i<PARNAME;i++) {\
    reservoirs[FrameClass::NAME].push_back(new CLASSNAME());\
    stacks[FrameClass::NAME].push_back(reservoirs[FrameClass::NAME].back());\
  }\
}

  make_reservoir_and_stack(basic, ctx.n_basic, BasicFrame);
  make_reservoir_and_stack(setup, ctx.n_setup, SetupFrame);
  // TODO
  
  /* reserve for all these..
  basic,     ///< data at payload
  
  avpkt,     ///< data at ffmpeg avpkt
  avframe,   ///< data at ffmpeg av_frame and ffmpeg av_codec_context
  
  yuvpbo,    ///< data at yuvpbo struct
  
  setup,     ///< setup data
  signal,    ///< signal to AVThread or OpenGLThread
  */
}


FrameFifo::~FrameFifo() {
  recycleAll();
  
#define delete_reservoir(NAME) {\
  Reservoir &reservoir=reservoirs[FrameClass::NAME];\
  for (auto it=reservoir.begin(); it!=reservoir.end(); ++it) { delete *it; };\
  }
  
  delete_reservoir(basic);
  delete_reservoir(setup);
  
  /*
  Reservoir &reservoir=reservoirs[FrameClass::basic]; // alias
  for (auto it=reservoir.begin(); it!=reservoir.end(); ++it) { delete *it; }
  }
  */
}


bool FrameFifo::writeCopy(Frame* f, bool wait) {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  
  // get an alias for the stack corresponding to this Frame
  
  /*
  try {
    Stack &stack = stacks.at(f->getFrameClass());
  }
  catch (std::out_of_range) {
    fifologger.log(LogLevel::normal) << "FrameFifo: "<<name<<" writeCopy: no stack for FrameClass "<<int(f->getFrameClass());
    return false;
  }
  */
  
  Stack &stack = stacks.at(f->getFrameClass());
  
  while (stack.empty()) { // deal with spurious wake-ups
    if (wait) {
      fifologger.log(LogLevel::normal) << "FrameFifo: "<<name<<" writeCopy: waiting for stack frames.  Frame="<<(*f)<<std::endl;
      this->ready_condition.wait(lk);
      // fifologger.log(LogLevel::normal) << "FrameFifo: "<<name<<" writeCopy: .. got stack frame.  " << std::endl;
    }
    else {
      fifologger.log(LogLevel::fatal) << "FrameFifo: "<<name<<" writeCopy: OVERFLOW! No more frames in stack.  Frame="<<(*f)<<std::endl;
      if (ctx.flush_when_full) {
        recycleAll();
      }
      return false;
    }
  }
  
  //std::cout << "FrameFifo : writeCopy : stack size0=" << stack.size() << std::endl;
  
  Frame *tmpframe=stack.front();  // .. the Frame* pointer to the Frame object is in reservoirs[FrameClass].  Frame*'s in stacks[FrameClass] are same Frame*'s as in reservoirs[FrameClass]
  stack.pop_front();              // .. remove that pointer from the stack
  
  //std::cout << "FrameFifo : writeCopy : stack size=" << stack.size() << std::endl;
  
  tmpframe->copyFrom(f);      // makes a copy with the correct typecast
  fifo.push_front(tmpframe);  // push_front takes a copy of the pointer // fifo: push: front, read: back
  
#ifdef FIFO_VERBOSE
  if (fifo.size()>1) {std::cout << "FrameFifo: "<<name<<" writeCopy: count=" << fifo.size() << std::endl;}
#endif
  
#ifdef TIMING_VERBOSE
  long int dt=(getCurrentMsTimestamp()-tmpframe->mstimestamp);
  if (dt>100) {
    std::cout << "FrameFifo: "<<name<<" writeCopy : timing : inserting frame " << dt << " ms late" << std::endl;
  }
#endif
  
  this->condition.notify_one(); // after receiving 
  return true;
}
  
  
Frame* FrameFifo::read(unsigned short int mstimeout) {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context  
  std::cv_status result;
  
#ifdef FIFO_VERBOSE
  // std::cout << "FrameFifo: "<<name<<"      read: mstimeout=" << mstimeout << std::endl;
#endif
  
  result=std::cv_status::no_timeout; // default, return no frame
  
  while(fifo.size()<=0) { // deal with spurious wakeups
    result=std::cv_status::no_timeout;
    if (mstimeout==0) {
      this->condition.wait(lk);
      // Once fifo size is <=0, we start waiting for the condition variable, i.e. for the "event"
      // So, what does condition.wait(lk) do?  Once we call it, lock is released ("atomically released" according to the reference pages).  And once the condition variable goes off, we reclaim the lock.
      // http://en.cppreference.com/w/cpp/thread/condition_variable
    }
    else {
#ifdef FIFO_VERBOSE
      // std::cout << "FrameFifo: "<<name<<" wait with mstimeout=" << mstimeout << std::endl;
#endif
      result=condition.wait_for(lk,std::chrono::milliseconds(mstimeout));
      break;
    }
  }
  
  if (result==std::cv_status::timeout) { // so, waiting expired
    return NULL;
  }
  
#ifdef FIFO_VERBOSE
  if (fifo.size()>1) {std::cout << "FrameFifo: "<<name<<" read: count=" << fifo.size() << std::endl;}
#endif
  Frame* tmpframe=fifo.back(); // fifo: push: front, read: back
  fifo.pop_back(); // remove the last element
  return tmpframe;
}


void FrameFifo::recycle_(Frame* f) {
  Stack &stack = stacks.at(f->getFrameClass());
  stack.push_back(f); // take: from the front.  recycle: to the back
}


void FrameFifo::recycle(Frame* f) {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  recycle_(f);
  ready_condition.notify_one();
}


void FrameFifo::recycleAll() { // move all frames from fifo to stack
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  auto it=fifo.begin();
  while (it!=fifo.end()) {
    recycle_(*it); // return to stack
    it=fifo.erase(it); // and erase from the fifo
  }
}


void FrameFifo::dumpStacks() {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  Stack stack;
  
  std::cout << "FrameFifo : dumpStacks : " << std::endl;
  for(auto it=stacks.begin(); it!=stacks.end(); ++it) {
    std::cout << "FrameFifo : dumpStacks : Stack=" << int(it->first) << std::endl;
    stack=it->second;
    for (auto its=stack.begin(); its!=stack.end(); ++its) {
      std::cout << "FrameFifo : dumpStacks :  " << *(*its) << std::endl;
    }
  }
  std::cout << "FrameFifo : dumpStacks : " << std::endl;
  
  
}
  
void FrameFifo::dumpFifo() {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  
  std::cout << "FrameFifo : dumpFifo : " << std::endl;
  for(auto it=fifo.begin(); it!=fifo.end(); ++it) {
    std::cout << "FrameFifo : dumpFifo : " << *(*it) << std::endl;
  }
  std::cout << "FrameFifo : dumpFifo : " << std::endl;
}
 
 
void FrameFifo::diagnosis() {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  Stack stack;
  
  std::cout << "FrameFifo : diagnosis : " << std::endl;
  std::cout << "FrameFifo : diagnosis : fifo  : " << fifo.size() << std::endl;
  std::cout << "FrameFifo : diagnosis : stack : ";
  for(auto it=stacks.begin(); it!=stacks.end(); ++it) {
    std::cout << int(it->first) << ":" << (it->second).size() << ", ";
  }
  std::cout << std::endl;
  std::cout << "FrameFifo : diagnosis : " << std::endl;
}
 
 
bool FrameFifo::isEmpty() {
  return fifo.empty();
}


