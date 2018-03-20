 /*
 * queues.cpp : Lockable "safe-queues" for multithreading use, for frame-queueing
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
 *  @file    queues.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.6 
 *  
 *  @brief Lockable safe-queues for frame queueing in multithreading applications
 */

#include "queues.h" // deque: http://en.cppreference.com/w/cpp/container/deque
#include "logging.h"

// WARNING: these define switches should be off (commented) by default
// #define FIFO_VERBOSE 1

void FrameFifo::initReservoir(std::vector<Frame> &reservoir, unsigned short n_stack) {
  int i;
  for(i=0; i<n_stack; i++) {
    reservoir.push_back(Frame());
  }
}


void FrameFifo::initPayload(std::vector<Frame> &reservoir) {
  for(auto it=reservoir.begin(); it!=reservoir.end(); ++it) {
   it->resize(int(DEFAULT_FRAME_FIFO_PAYLOAD_SIZE));
  }
}


void FrameFifo::initStack(std::vector<Frame> &reservoir, std::deque<Frame*> &stack) {
  int i;
  for(i=0; i<reservoir.size(); i++) {
    stack.push_front(&reservoir[i]); // pointer to an element in the reservoir
  }
}


void FrameFifo::dump(std::deque<Frame*> &queue) {
  int i=0;  
  for (auto it = queue.begin(); it != queue.end(); ++it) {
    queuelogger.log(LogLevel::normal) << i << " : "<< **it << std::endl;
    ++i;
  }
}


FrameFifo::FrameFifo(const char* name, unsigned short int n_stack, bool clear_when_filled) : name(name), n_stack(n_stack), clear_when_filled(clear_when_filled), count(0) {
  unsigned short int i;
  
  // reservoir.resize(this->n_stack); // avoid resizing vectors and deques of pointers ..
  // stack.resize(this->n_stack);
  
  for(i=0; i<(this->n_stack); i++) {
    reservoir.push_back(Frame());
    // reservoir[i].setMsTimeStamp(i); // checking ..
    // reservoir.back().reserve(1024*1024); 
    // stack.push_front(&(reservoir.back())); // insert pointers into the stack // this won't work! .. segfaults, valgrind errors, why?
    //// .. .back() should return a reference to the last pointer
  }
  
  // /* // this version works
  for(i=0; i<(this->n_stack); i++) {
    reservoir[i].reserve(int(DEFAULT_FRAME_FIFO_PAYLOAD_SIZE));
    stack.push_front(&reservoir[i]); // pointer to an element in the reservoir
  }
  // */
  
  
  // stack.push_front(&reservoir.front());
  
  /*
  for (std::vector<Frame>::iterator it = reservoir.begin(); it != reservoir.end(); ++it) {
    it->reserve(1024*1024);
    stack.push_front(&it); // insert pointers from reservoir into the stack
  }
  */
  
  // this->ready_condition.notify_one();

}


FrameFifo::~FrameFifo() {
  // When this goes out of scope ..
  queuelogger.log(LogLevel::crazy) << "FrameFifo: "<<name<<" destructor!" << std::endl;
  // reservoir (vector of frames gets cleaned up)
  // stack.clear(); // no need for this .. stack is emptied automatically
  // fifo.clear(); // no need for this either ..
  queuelogger.log(LogLevel::crazy) << "FrameFifo: "<<name<<" bye from destructor" << std::endl;
}


Frame* FrameFifo::getFrame() {
  Frame* tmpframe;
  
  if (this->stack.empty()) {
    queuelogger.log(LogLevel::fatal) << "FrameFifo: "<<name<<" getFrame: OVERFLOW! No more frames in stack "<<std::endl;
    dumpFifo();
    dumpStack();
    return NULL;
  }
  
  // tmpframe=this->stack[0]; // take a pointer to frame from the pre-allocated stack
  tmpframe=this->stack.front();
  this->stack.pop_front(); // .. remove that pointer from the stack
  
  return tmpframe;
}


bool FrameFifo::writeCopy(Frame* f, bool wait) { // take a frame from the stack, copy contents of f into it and insert the copy into the beginning of the fifo ("copy-on-write")
  Frame* tmpframe;
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context

  ///*
  while (this->stack.empty()) { // deal with spurious wake-ups
    if (wait) {
      queuelogger.log(LogLevel::normal) << "FrameFifo: "<<name<<" writeCopy: waiting for stack frames.  Frame="<<(*f)<<std::endl;
      this->ready_condition.wait(lk);
      // queuelogger.log(LogLevel::normal) << "FrameFifo: "<<name<<" writeCopy: .. got stack frame.  " << std::endl;
    }
    else {
      queuelogger.log(LogLevel::fatal) << "FrameFifo: "<<name<<" writeCopy: OVERFLOW! No more frames in stack.  Frame="<<(*f)<<std::endl;
      if (clear_when_filled) {
        recycleAll();
      }
      return false;
    }
  }
  //*/
  
  /*
  if (this->stack.empty()) {
    queuelogger.log(LogLevel::fatal) << "FrameFifo: "<<name<<" writeCopy: OVERFLOW! No more frames in stack.  Frame="<<(*f)<<std::endl;
    return false;
  }
  */
  
  // tmpframe=this->stack[0]; // take a pointer to frame from the pre-allocated stack
  tmpframe=this->stack.front();
  this->stack.pop_front(); // .. remove that pointer from the stack
  
  /* this is done by the copy-assignment down there..
  // at this stage, resize the used stack frame to target size
  if (tmpframe->payload.capacity()<target_size) { // if frame payload not large enough, make it bigger
    queuelogger.log(LogLevel::debug) << "FrameFifo: "<<name<<" writeCopy: resizing stack element from "<< tmpframe->payload.capacity() << " to "<< target_size <<std::endl;
    tmpframe->payload.reserve(target_size);
  }
  // .. we could additionally run through the stack and resize all elements there
  */
  
  *(tmpframe)=*(f); // a copy-assignment of objects (not pointers) makes a deep copy in our case (see the comments in the Frame class) .. actually, this should resize the payload capacity as well..
  this->fifo.push_front(tmpframe);
  ++(this->count);
  
#ifdef FIFO_VERBOSE
  if (count>1) {std::cout << "FrameFifo: "<<name<<" writeCopy: count=" << this->count << std::endl;}
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


// pop a frame from the end of the fifo and return the frame
Frame* FrameFifo::read(unsigned short int mstimeout) {
  Frame* tmpframe;
  std::cv_status result;
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  
  /// this->ready_condition.notify_one();
  
#ifdef FIFO_VERBOSE
  // std::cout << "FrameFifo: "<<name<<"      read: mstimeout=" << mstimeout << std::endl;
#endif
  
  result=std::cv_status::no_timeout; // default, return no frame
  
  while((this->count)<=0) { // deal with spurious wakeups
    result=std::cv_status::no_timeout;
    if (mstimeout==0) {
      this->condition.wait(lk);
      // this->condition.wait(lk); // so, once the count is <=0, we start waiting for the condition variable, i.e. for the "event"
      // So, what does condition.wait(lk) do?  Once we call it, lock is released ("atomically released" according to the reference pages).  And once the condition variable goes off, we reclaim the lock.
      // http://en.cppreference.com/w/cpp/thread/condition_variable
    }
    else {
#ifdef FIFO_VERBOSE
      // std::cout << "FrameFifo: "<<name<<" wait with mstimeout=" << mstimeout << std::endl;
#endif
      result=this->condition.wait_for(lk,std::chrono::milliseconds(mstimeout));
      break;
    }
  }
  
  if (result==std::cv_status::timeout) { // so, waiting expired
    return NULL;
  }
  
#ifdef FIFO_VERBOSE
  if (count>1) {std::cout << "FrameFifo: "<<name<<"      read: count=" << this->count << std::endl;}
#endif
  // tmpframe=this->fifo[this->count-1]; // take the last element
  tmpframe=fifo.back();
  this->fifo.pop_back(); // remove the last element
  --(this->count);
  return tmpframe;
}


void FrameFifo::recycle(Frame* f) {
  std::unique_lock<std::mutex> lk(this->mutex);
  target_size=std::max(target_size,f->payload.capacity()); // update target_size 
  f->reset();
  f->payload.reserve(target_size);  //  "If new_cap is greater than the current capacity(), new storage is allocated, otherwise the method does nothing."
  stack.push_front(f); 
  this->ready_condition.notify_one(); // TODO: release wait lock
  
  /*
  std::cout << "FrameFifo: recycle: dumpStack:" << std::endl;
  dumpStack();
  std::cout << "FrameFifo: recycle: dumpStack:" << std::endl;
  */
}


void FrameFifo::recycleAll() { // recycle all frames from fifo back to stack
  std::unique_lock<std::mutex> lk(this->mutex);
  Frame *tmpframe;
  queuelogger.log(LogLevel::debug) << "FrameFifo: recycleAll: " << std::endl;
  while(fifo.size()>0) {
    // tmpframe=this->fifo[this->count-1]; // take the last element
    tmpframe=fifo.back();
    stack.push_front(tmpframe);
    fifo.pop_back();
    --(this->count); // aux counter .. do we really need this..?
  }
}
    

/*
// pop a frame from the end of the fifo, recycle it into stack and return a copy of the frame ("copy-on-read")
Frame* FrameFifo::readCopy() {
}
*/

void FrameFifo::dumpStack() {
  unsigned short int i=0;
  
  std::cout << "FrameFifo: "<<name<<" dumpStack> target_size="<< target_size << std::endl;
  for (std::deque<Frame*>::iterator it = stack.begin(); it != stack.end(); ++it) {
    std::cout << i << " : "<< **it << std::endl;
    ++i;
  }
  std::cout << "FrameFifo: "<<name<<" <dumpStack" << std::endl;
}


void FrameFifo::dumpFifo() {
  std::cout << "FrameFifo: "<<name<<" dumpFifo>" << std::endl;
  for (std::deque<Frame*>::iterator it = fifo.begin(); it != fifo.end(); ++it) {
    std::cout << **it << std::endl;
  }
  std::cout << "FrameFifo: "<<name<<" <dumpFifo" << std::endl;
}


void FrameFifo::diagnosis() {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  std::cout << "FrameFifo: " << name << " diagnosis: FIFO: " << fifo.size() << " STACK: " << stack.size() << std::endl;
}


bool FrameFifo::isEmpty() {
  std::unique_lock<std::mutex> lk(this->mutex);
  return this->fifo.empty();
}



/*
void FrameFifo::waitAvailable() {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  std::cout << "FrameFifo : waitAvailable : waiting .." << std::endl;
  ready_condition.wait(lk);
  std::cout << "FrameFifo : waitAvailable : .. waited" << std::endl;
}
*/


FifoFrameFilter::FifoFrameFilter(const char* name, FrameFifo& framefifo) : FrameFilter(name), framefifo(framefifo) {
};

void FifoFrameFilter::go(Frame* frame) {
  framefifo.writeCopy(frame);
}



BlockingFifoFrameFilter::BlockingFifoFrameFilter(const char* name, FrameFifo& framefifo) : FrameFilter(name), framefifo(framefifo) {
};

void BlockingFifoFrameFilter::go(Frame* frame) {
  // std::cout << "BlockingFifoFrameFilter: go" << std::endl;
  framefifo.writeCopy(frame,true);
  /*
  
  condition variable A: triggered when there are frames in the queue
  condition variable B: triggered when there are frames available in the stack
  
  this->mutex controls locking with "lk"
  
  server:
  framefifo.writeCopy     reserves mutex .. uses condition_variable.notify_one(lk) 
  
  
  server with wait:
  
  framefifo.writeCopy
                          reserve mutex .. observe the state of the fifo .. if no more frames available in the stack, go to a waiting state .. wait for a condition variable B : condition_variable.wait(lk) releases the lock every now then
    
  client:
  framefifo.read          reserves mutex, trigger condition variable B.  Wait for condition variable A.
  
  
  waits for a condition variable.   condition_variable.wait(lk) releases the mutex for an instant .. imagine that it releases it every now and then ..
  
  
  ----
  
  server:
  framefifo.writeCopy    reserves mutex
  

  */
}


