#ifndef decoderthread_HEADER_GUARD
#define decoderthread_HEADER_GUARD
/*
 * decoderthread.h :
 * 
 * (c) Copyright 2017-2024 Sampsa Riikonen
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
 *  @file    decoderthread.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.6.1 
 *  
 *  @brief
 */ 

#include "constant.h"
#include "frame.h"
#include "thread.h" 
#include "decoder.h"
#include "tools.h"
#include "framefilter.h"

class DecoderThread : public Thread { // <pyapi>
  
public: // <pyapi>
    /** Default constructor
    * 
    * @param name              Name of the thread
    * @param outfilter         Outgoing frames are written here.  Outgoing frames may be of type FrameType::avframe
    * @param fifo_ctx          Parametrization of the internal FrameFifo
    * 
    */
    DecoderThread(const char* name, FrameFilter& outfilter, FrameFifoContext fifo_ctx=FrameFifoContext()); // <pyapi>
    virtual ~DecoderThread(); ///< Default destructor.  Calls AVThread::stopCall                                   // <pyapi>
    
protected: // frame input
    FrameFifo               infifo;           ///< Incoming frames are read from here
    FifoFrameFilter         infilter;         ///< Write incoming frames here
    BlockingFifoFrameFilter infilter_block;   ///< Incoming frames can also be written here.  If stack runs out of frames, writing will block
    int                     n_threads;
    
protected:
    FrameFilter& outfilter;                ///< Outgoing, decoded frames are written here
    std::vector<Decoder*> decoders;        ///< A vector/list of registered and instantiated decoders
    long int     mstimetolerance;          ///< Drop frames if they are in milliseconds this much late
    AbstractFileState state;               ///< Seek, play, stop or what
    std::vector<SetupFrame> setupframes;   ///< Save decoder(s) setup information

private: // framefilter for chaining output for outfilter2
    TimestampFrameFilter2 timefilter;
    bool use_time_correction;

protected:
    bool is_decoding; ///< should currently decode or not
        
protected: // Thread member redefinitions
    std::deque<AVSignalContext> signal_fifo;   ///< Redefinition of signal fifo.

protected:
    virtual Decoder* chooseAudioDecoder(AVCodecID codec_id);
    virtual Decoder* chooseVideoDecoder(AVCodecID codec_id); ///< Chooses a video decoder
    virtual Decoder* fallbackAudioDecoder(AVCodecID codec_id);
    /** If the the video decoder obtained from chooseVideoDecoder fails for some reason,
     * provide a fallback video decoder instead.
     */
    virtual Decoder* fallbackVideoDecoder(AVCodecID codec_id);

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
    /** Set number of decoding threads
     * 
     * - Must be called before the thread is run
     * 
     */
    void setTimeCorrection(bool val);             // <pyapi>
    FifoFrameFilter &getFrameFilter();            // <pyapi>
    FifoFrameFilter &getBlockingFrameFilter();    // <pyapi>
    void setTimeTolerance(long int mstol);    ///< API method: decoder will scrap late frames that are mstol milliseconds late.  Call before starting the thread. // <pyapi>
    void setNumberOfThreads(int n_threads);       // <pyapi>
    void decodingOnCall();   ///< API method: enable decoding        // <pyapi>
    void decodingOffCall();  ///< API method: pause decoding         // <pyapi>
    void requestStopCall();  ///< API method: Like Thread::stopCall() but does not block. // <pyapi>
}; // <pyapi>


#endif
