/*
 * testthread.cpp : Launch a cpp thread from python, give that thread a callback that's called by the thread
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
 *  @file    testthread.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.13.1 
 *  
 *  @brief Launch a cpp thread from python, give that thread a callback that's called by the thread
 */ 

#include "testthread.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for; 


// https://stackoverflow.com/questions/16606872/calling-python-method-from-c-or-c-callback/16609899#16609899
// This is a very nice answer:
// https://stackoverflow.com/questions/37513957/does-python-gil-need-to-be-taken-care-when-work-with-multi-thread-c-extension

TestThread::TestThread(const char* name) : Thread(name), pyfunc(NULL) {
}

TestThread::~TestThread() {
  if (pyfunc!=NULL) {
    Py_DECREF(pyfunc);
  }
}

void TestThread::preRun() {
}

void TestThread::postRun() {}

void TestThread::setCallback(PyObject* pobj) {
  // pass here, say "signal.emit" or a function/method that accepts single argument
  if (PyCallable_Check(pobj)) { // https://docs.python.org/3/c-api/type.html#c.PyTypeObject
    Py_INCREF(pobj);
    pyfunc=pobj;
  }
  else {
    std::cout << "TestThread: setCallback: needs python callable" << std::endl;
    pyfunc=NULL;
  }
}

void TestThread::triggerCallback(int i) {
  // GIL needs to be acquired
  // The problem is, that closeThread (i.e., pthread_join) blocks and the GIL is never released from there ..
  
  if (pyfunc!=NULL) {
    PyGILState_STATE gstate;
#ifdef TESTTHREAD_VERBOSE
    std::cout << "  TestThread: triggerCallback: getting GIL" << std::endl;
#endif
    gstate = PyGILState_Ensure();
#ifdef TESTTHREAD_VERBOSE
    std::cout << "  TestThread: triggerCallback: got GIL" << std::endl;
#endif
    
    /*
    PyObject *argList =Py_BuildValue("i", i); // https://docs.python.org/3.5/c-api/arg.html#c.Py_BuildValue
    if (!argList) {
      std::cout << "  TestThread: could not create argList" << std::endl;
    }
    */
#ifdef TESTTHREAD_VERBOSE
    std::cout << "  TestThread: triggerCallback: calling pyfunc" << std::endl;
#endif
    
    // PyObject *res =PyObject_CallObject(pyfunc, argList); // does not work..?  Is this just for the __call__ method?
    PyObject_CallFunction(pyfunc, "i", i); // WORKS!
    
    // PyObject *res =PyObject_CallObject(pyfunc, NULL);
#ifdef TESTTHREAD_VERBOSE
    std::cout << "  TestThread: triggerCallback: called pyfunc" << std::endl;
#endif
    
    // Py_DECREF(argList);
    // Py_DECREF(res); // nopes
    
    PyGILState_Release(gstate);
  }
}


void TestThread::run() { // this is running, and has nothing to do with the GIL
  loop=true;
  int i=0;
  
  while(loop) {
    sleep_for(1s);
#ifdef TESTTHREAD_VERBOSE
    std::cout << "TestThread: run: calling triggerCallback with " << i << std::endl;
#endif
    triggerCallback(i); // if pthread_join called, this hangs..! .. that's because main thread is stuck
    // .. and we can't get GIL from it, right?
#ifdef TESTTHREAD_VERBOSE
    std::cout << "TestThread: run: called triggerCallback" << std::endl;
#endif
    i++;
    {
      std::unique_lock<std::mutex> lk(this->mutex);
      // process pending signals..
      for (auto it = signal_fifo.begin(); it != signal_fifo.end(); ++it) {
#ifdef TESTTHREAD_VERBOSE
        std::cout << "                          "<< this->name <<" processed signal " << std::endl;
#endif
        if (it->signal==TestSignal::exit) {loop=false;}
        else if (it->signal==TestSignal::add) {i+=100;}
      }
      signal_fifo.clear();
    }
  }
#ifdef TESTTHREAD_VERBOSE
  std::cout << "TestThread: run: bye!" << std::endl;
#endif
}


void TestThread::sendSignal(TestSignalContext signal_ctx) {
  std::unique_lock<std::mutex> lk(this->mutex);
  this->signal_fifo.push_back(signal_ctx);
}


void TestThread::stopCall() { // remember: this is a frontend method.  When called from python, it acquires GIL
  // btw: we are not making any python calls here, so we should release GIL
  if (!this->has_thread) {return;}
  TestSignalContext signal_ctx;
  signal_ctx.signal=TestSignal::exit;
  sendSignal(signal_ctx);
  
  // https://stackoverflow.com/questions/5159040/is-it-possible-to-release-the-gil-before-a-c-function-that-blocks-and-might-call
  Py_BEGIN_ALLOW_THREADS;
  this->closeThread();
  Py_END_ALLOW_THREADS;
  
  this->has_thread=false;
}


void TestThread::addCall() {
  TestSignalContext signal_ctx;
  
  signal_ctx.signal=TestSignal::add;
  sendSignal(signal_ctx);
}

  
