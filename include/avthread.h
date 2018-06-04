#ifndef AVTHREAD_HEADER_GUARD 
#define AVTHREAD_HEADER_GUARD
/*
 * avthread.h : FFmpeg decoding thread
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
 *  @file    avthread.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.4.6 
 *  
 *  @brief FFmpeg decoding thread
 *
 */

#include "constant.h"
#include "frame.h"
#include "thread.h" 
#include "decoder.h"
#include "tools.h"
#include "framefilter.h"



/** A thread consuming frames and feeding them to various encoders
 * 
 * This class implements a "consumer & producer" thread: it reads Frame instances from a FrameFifo, decodes them and writes them to an outbound FrameFilter
 * 
 * Each arriving frame is inspected for its subsession index, i.e. for Frame.subsession_index and passed on to the adequate decoder, which are warehoused in AVThread::decoders.  This vector/list is initialized by sending special "setup frames" to AVThread.  These frames correspond to frametype FrameType::setup, and they are typically sent by the LiveThread after an rtsp negotiation
 * 
 * See also \ref pipeline
 * 
 * @ingroup decoding_tag
 * @ingroup threading_tag
 */
class AVThread : public Thread { // <pyapi>
  

public: // <pyapi>
  /** Default constructor
   * 
   * @param name              Name of the thread
   * @param outfilter         Outgoing frames are written here.  Outgoing frames may be of type FrameType::avframe
   * @param fifo_ctx          Parametrization of the internal FrameFifo
   * 
   */
  AVThread(const char* name, FrameFilter& outfilter, FrameFifoContext fifo_ctx=FrameFifoContext()); // <pyapi>
  ~AVThread(); ///< Default destructor.  Calls AVThread::stopCall                                   // <pyapi>
  
protected: // frame input
  FrameFifo               infifo;           ///< Incoming frames are read from here
  FifoFrameFilter         infilter;         ///< Write incoming frames here
  BlockingFifoFrameFilter infilter_block;   ///< Incoming frames can also be written here.  If stack runs out of frames, writing will block
  
protected:
  FrameFilter& outfilter;               ///< Outgoing, decoded frames are written here
  std::vector<Decoder*> decoders;   ///< A vector/list of registered and instantiated decoders
  long int     mstimetolerance;         ///< Drop frames if they are in milliseconds this much late
  
protected:
  bool is_decoding; ///< should currently decode or not
  
protected: // Thread member redefinitions
  std::deque<AVSignalContext> signal_fifo;   ///< Redefinition of signal fifo.
  
public: // redefined virtual functions
  void run();
  void preRun();
  void postRun();
  void sendSignal(AVSignalContext signal_ctx); ///< Redefined : Thread::SignalContext has been changed to AVThread::SignalContext
  
protected: 
  FrameFifo &getFifo();
  
protected:
  void handleSignals();
  
public: // API <pyapi>
  FifoFrameFilter &getFrameFilter();            // <pyapi>
  FifoFrameFilter &getBlockingFrameFilter();    // <pyapi>
  void setTimeTolerance(long int mstol);    ///< API method: decoder will scrap late frames that are mstol milliseconds late.  Call before starting the thread. // <pyapi>
  void decodingOnCall();   ///< API method: enable decoding        // <pyapi>
  void decodingOffCall();  ///< API method: pause decoding         // <pyapi>
  void stopCall();         ///< API method: terminates the thread  // <pyapi>
}; // <pyapi>

#endif
