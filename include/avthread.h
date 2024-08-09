#ifndef AVTHREAD_HEADER_GUARD 
#define AVTHREAD_HEADER_GUARD
/*
 * avthread.h : FFmpeg decoding thread
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
 *  @file    avthread.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.6.1 
 *  
 *  @brief FFmpeg decoding thread
 *
 */

#include "decoderthread.h"

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
class AVThread : public DecoderThread { // <pyapi>
  
public: // <pyapi>
    /** Default constructor
    * 
    * @param name              Name of the thread
    * @param outfilter         Outgoing frames are written here.  Outgoing frames may be of type FrameType::avframe
    * @param fifo_ctx          Parametrization of the internal FrameFifo
    * 
    */
    AVThread(const char* name, FrameFilter& outfilter, FrameFifoContext fifo_ctx=FrameFifoContext());   // <pyapi>
    virtual ~AVThread(); ///< Default destructor.  Calls AVThread::stopCall                             // <pyapi>

}; // <pyapi>

#endif
