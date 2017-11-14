/*
 * threads.cpp : Convenience classes for multithreading
 * 
 * Copyright 2017 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Valkka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Valkka.  If not, see <http://www.gnu.org/licenses/>. 
 * 
 */

/** 
 *  @file    threads.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
 *  
 *  @brief A convenience class for multithreading, similar to Python's standard library "threading.Thread"
 *
 *  @section DESCRIPTION
 *  
 *  Yes, the description
 *
 */ 

#include "threads.h"
#include "logging.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for; 


Thread::Thread(const char* name, int core_id) : name(std::string(name)), core_id(core_id), has_thread(false), loop(false) {
  // this->name       =std::string(name);
  // this->has_thread=false;
  // this->signal    =Thread::Signals::none;
  int npros=sysconf(_SC_NPROCESSORS_CONF);
  
  if (core_id>=npros) {   // https://www.gnu.org/software/libc/manual/html_node/Processor-Resources.html
    threadlogger.log(LogLevel::fatal) << "Thread: constructor: WARNING: core_id="<< core_id <<" but number of processors is " << npros << this->name <<std::endl;
    core_id=0;
  }
  start_mutex.lock();
}


Thread::~Thread() {
  if (this->has_thread) {
    threadlogger.log(LogLevel::debug) << "Thread: destructor: joining "<< this->name <<std::endl;
    this->closeThread();
    threadlogger.log(LogLevel::debug) << "Thread destructor: joined "<< this->name <<std::endl;
  }
}


void Thread::mainRun() {// for std::thread version
  this->preRun();
  this->run(); // unlock start_mutex here
  this->postRun();
  threadlogger.log(LogLevel::debug) << "Thread: mainRun: bye from "<< this->name <<std::endl;
}

#ifdef STD_THREAD
#else
void* Thread::mainRun_(void *p) {// for the pthread_* version
  ( (Thread*)p )->mainRun();
  // this->mainRun();
  return NULL;
}
#endif


void Thread::closeThread() {
  #ifdef STD_THREAD
  // std::thread way
  this->internal_thread.join();
#else
  // pthread_* way
  void *res;
  int i;
  i=pthread_join(internal_thread, &res);
  if (i!=0) {perror("Thread: closeThread: WARNING! join failed"); exit(1);}
  free(res); // free resources allocated by thread
  i=pthread_attr_destroy(&thread_attr);
  if (i!=0) {perror("Thread: closeThread: WARNING! pthread_attr_destroy failed"); exit(1);}
#endif  
}

void Thread::startCall() {
  this->has_thread=true;
  
#ifdef STD_THREAD
  // std::thread way
  this->internal_thread = std::thread([=] { mainRun();});
#else  
  int i;
  
  i=pthread_attr_init(&thread_attr);
  if (i!=0) {perror("Thread: startCall: WARNING! pthread_attr_init failed"); exit(1);}
  
  if (core_id>-1) {
    threadlogger.log(LogLevel::debug) << "Thread: startCall: binding thread "<< this->name <<" to processor "<< core_id<< std::endl;
    CPU_ZERO(&cpuset); // http://man7.org/linux/man-pages/man3/CPU_SET.3.html
    CPU_SET(core_id, &cpuset);
    
    // number of processes: // https://stackoverflow.com/questions/150355/programmatically-find-the-number-of-cores-on-a-machine
    // sysconf (_SC_NPROCESSORS_CONF)
    
    // pthread_* way:
    // http://man7.org/linux/man-pages/man3/pthread_attr_init.3.html
    // 
    //int pthread_attr_init(pthread_attr_t *attr);
    //int pthread_attr_setaffinity_np(pthread_attr_t *attr, size_t cpusetsize, const cpu_set_t *cpuset); // cpusetsize is the length of cpuset
    //int pthread_create(pthread_t *thread, const pthread_attr_t *attr,  void *(*start_routine) (void *), void *arg);
    i=pthread_attr_setaffinity_np(&thread_attr,sizeof(cpu_set_t),&cpuset); // cpusetsize is the length of cpuset
    if (i!=0) {perror("Thread: startCall: WARNING! pthread_attr_setaffinity_np failed"); exit(1);}
  }
    
  i=pthread_create(&internal_thread, &thread_attr, this->mainRun_, this);
  if (i!=0) {perror("Thread: startCall: WARNING! could not create thread"); exit(1);}
#endif
  
  threadlogger.log(LogLevel::debug) << "Thread: startCall: waiting for "<< this->name << " to start"<<std::endl;
  this->start_mutex.lock();
}


void Thread::stopCall() {
  SignalContext signal_ctx;
  signal_ctx.signal=Signals::exit;
  this->sendSignal(signal_ctx);
  this->closeThread();
  this->has_thread=false;
}


void Thread::sendSignal(SignalContext signal_ctx) {
  std::unique_lock<std::mutex> lk(this->mutex);
  // this->signal=signal;
  this->signal_fifo.push_back(signal_ctx);  
}


void Thread::sendSignalAndWait(SignalContext signal_ctx) {
  std::unique_lock<std::mutex> lk(this->mutex);
  // this->signal=signal;
  this->signal_fifo.push_back(signal_ctx); 
  this->condition.wait(lk);
}



TestProducerThread::TestProducerThread(const char* name, FrameFifo* framefifo, int index) : Thread(name), framefifo(framefifo), index(index) {
}

void TestProducerThread::preRun() {}
void TestProducerThread::postRun() {}
void TestProducerThread::run() {
  uint i;
  Frame f;
  bool res;
  
  start_mutex.unlock();
  for(i=0; i<5; i++) {
    f.setMsTimestamp(i+index*1000);
    threadlogger.log(LogLevel::normal) << this->name <<" writeCopy " << i << " writing " << f << std::endl;
    res=framefifo->writeCopy(&f);
    threadlogger.log(LogLevel::normal) << this->name <<" writeCopy " << i << " returned " << res << std::endl;
    // framefifo->dumpFifo();
    sleep_for(0.1s);
  }
  
}

  
TestConsumerThread::TestConsumerThread(const char* name, FrameFifo* framefifo) : Thread(name), framefifo(framefifo) {
}

void TestConsumerThread::preRun() {}
void TestConsumerThread::postRun() {}
void TestConsumerThread::run() {
  Frame* f;
  SignalContext signal_ctx;
  // bool ok;
  
  start_mutex.unlock();
  f=NULL;
  // ok=true;
  loop=true;
  while(loop) {
    // f=framefifo->read(0);   // get frame from fifo
    f=framefifo->read(Timeouts::thread);
    if (!f) {
      threadlogger.log(LogLevel::normal) << "                          "<< this->name <<" timeout expired!" << std::endl;
      }
    else {
      threadlogger.log(LogLevel::normal) << "                          "<< this->name <<" got frame "<<*f << std::endl;
      // do something with the frame   .. say, copy data to ffmpeg avpacket
      // framefifo->recycle(f); // return it to the stack once you're done
      sleep_for(0.2s);
      threadlogger.log(LogLevel::normal) << "                          "<< this->name <<" hassled with the frame" << std::endl;
      framefifo->recycle(f);
      framefifo->dumpStack();
      }
    
    {
      std::unique_lock<std::mutex> lk(this->mutex);
      // process pending signals..
      for (std::deque<SignalContext>::iterator it = signal_fifo.begin(); it != signal_fifo.end(); ++it) { // it == pointer to the actual object (struct SignalContext)
        threadlogger.log(LogLevel::normal) << "                          "<< this->name <<" processed signal " << std::endl;
        if (it->signal==Thread::Signals::exit) { loop=false; }
      }
      signal_fifo.clear();
    }
    
  }
}

