#ifndef testthread_HEADER_GUARD
#define testthread_HEADER_GUARD
/*
 * testthread.h :
 * 
 * Copyright 2017-2023 Valkka Security Ltd. and Sampsa Riikonen
 * Copyright 2024 Sampsa Riikonen
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
 *  @file    testthread.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.6.1 
 *  
 *  @brief
 */ 

#include "Python.h"
#include "thread.h"


enum class TestSignal {
  none,
  exit,
  add
};



struct TestSignalContext {
  TestSignal  signal;
};



class TestThread : public Thread {                       // <pyapi>
  
public:                                                  // <pyapi>
  TestThread(const char* name);                          // <pyapi>
  virtual ~TestThread();                                 // <pyapi>
  
protected:
  std::deque<TestSignalContext> signal_fifo;    ///< Redefinition of signal fifo (Thread::signal_fifo becomes hidden)
  
protected:
  PyObject *pyfunc;
  
public:
  void run();
  void preRun();
  void postRun();

protected:
  void triggerCallback(int i);
  void sendSignal(TestSignalContext signal_ctx);
  
public:                                                 // <pyapi>
  void setCallback(PyObject* pobj);                     // <pyapi>
  void stopCall();                                      // <pyapi>
  void addCall();                                       // <pyapi>
};                                                      // <pyapi>


#endif
