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
 *  @version 0.2.0 
 *  
 *  @brief FFmpeg decoding thread
 *
 */

#include "sizes.h"
#include "frames.h"
#include "threads.h" 
#include "decoders.h"


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
  
public:

  /** Redefinition of characteristic signals for AVThread
   * 
   */
  enum class Signals {
    none,
    exit,
    on,  ///< turn decoding on 
    off  ///< turn decoding off
  };

  
  /** Redefinition of characteristic signal contexts (info that goes with the signal) for AVThread thread
   *
   * remember : Thread::SignalContext is not overwritten, only hidden
   * 
   */
  struct SignalContext {
    Signals signal;
    // AVConnectionContext connection_context; // in the case we want pass more information
  };


public: // <pyapi>
  /** Default constructor
   * 
   * @param name      Name of the thread
   * @param infifo    Incoming frames are consumed from here
   * @param outfilter Outgoing frames are written here.  Outgoing frames may be of type FrameType::avframe
   * 
   */
  AVThread(const char* name, FrameFifo& infifo, FrameFilter& outfilter, int core_id=-1); // <pyapi>
  ~AVThread(); ///< Default destructor // <pyapi>
  
protected:
  FrameFifo& infifo;                    ///< Incoming frames are read from here
  FrameFilter& outfilter;               ///< Outgoing, decoded frames are written here
  std::vector<DecoderBase*> decoders;   ///< A vector/list of registered and instantiated decoders
  
protected:
  bool is_decoding; ///< should currently decode or not
  
protected: // Thread member redefinitions
  std::deque<SignalContext> signal_fifo;   ///< Redefinition of signal fifo.  Signal fifo of Thread::SignalContext(s) is now hidden.
  
public: // redefined virtual functions
  void run();
  void preRun();
  void postRun();
  void sendSignal(SignalContext signal_ctx); ///< Must be explicitly *redefined* just in case : Thread::SignalContext has been changed to AVThread::SignalContext  

protected:
  /*! @copydoc Thread::hangleSignals
   */
  void handleSignals();
  
public: // API <pyapi>
  void decodingOnCall();   ///< API method: enable decoding        // <pyapi>
  void decodingOffCall();  ///< API method: pause decoding         // <pyapi>
  void stopCall();         ///< API method: terminates the thread  // <pyapi>
}; // <pyapi>

#endif 