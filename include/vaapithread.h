#ifndef vaapithread_HEADER_GUARD
#define vaapithread_HEADER_GUARD
/*
 * vaapithread.h :
 * 
 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
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
 *  @file    vaapithread.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.5.2 
 *  
 *  @brief
 */ 

#include "decoderthread.h"
#include "hwdecoder.h"

class VAAPIThread : public DecoderThread { // <pyapi>
  
public: // <pyapi>
    /** Default constructor
    * 
    * @param name              Name of the thread
    * @param outfilter         Outgoing frames are written here.  Outgoing frames may be of type FrameType::avframe
    * @param fifo_ctx          Parametrization of the internal FrameFifo
    * 
    */
    VAAPIThread(const char* name, FrameFilter& outfilter, FrameFifoContext fifo_ctx=FrameFifoContext());   // <pyapi>
    virtual ~VAAPIThread(); ///< Default destructor.  Calls AVThread::stopCall                             // <pyapi>

protected:
    virtual Decoder* chooseAudioDecoder(AVCodecID codec_id);
    virtual Decoder* chooseVideoDecoder(AVCodecID codec_id);
    virtual Decoder* fallbackAudioDecoder(AVCodecID codec_id);
    virtual Decoder* fallbackVideoDecoder(AVCodecID codec_id);

}; // <pyapi>


#endif
