#ifndef filestream_HEADER_GUARD
#define filestream_HEADER_GUARD
/*
 * filestream.h :
 * 
 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
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
 *  @file    filestream.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.6.1 
 *  
 *  @brief
 */ 

#include "frame.h"
#include "thread.h"
#include "framefilter.h"
#include "logging.h"
#include "tools.h"



/** A general class for on-disk stored streams
 * 
 * This class keeps the books for each file stream, in particular:
 *
 * - Desider target time (FileStream::target_mstimestamp_)
 * - Timestamp of the previous frame (FileStream::frame_mstimestamp_)
 * - State of the stream (FileStream::state)
 * 
 * Subclasses:
 * 
 * - For cached streams, should maintain cache
 * - For FFmpeg/libAV etc. streams, should maintain the file handles
 * 
 * In variable names, underscore means stream time.  See \ref timing
 * 
 * Threads use the (derived) class as follows:
 * 
 * - Thread should be generic / dummy
 * - Reads frames from a FrameFifo and feeds them to the AbstractFileStream
 * - At the same time "runs" the AbstractFileStream
 * - Passes commands to the AbstractFileStream (seek, play, stop, etc.)
 * 
 * 
 * while (loop):
 * 
 *      if ( (next_pts - mstime) <= 0)
 * 
 *          next_pts = AbstractFileStream->update(mstime)
 *              --> if called too early, returns
 *              => next_pts = AbstractFileStream->pullNextFrame()
 *                  => writes frame into FrameFilter
 *                  => take next time frame .. if frames with same timestamp, call recursively
 *                  => return next_pts
 *                  => set internal state : self.next_frame
 *          --> next_pts in wallclock time
 *          --> when seeking, next_pts is smaller than mstime for many consecutive frames  
 * 
 *      mstime = getCurrentMsTimestamp()
 *      --> handle signals if enough time has passed
 * 
 *      (refresh mstime)
 * 
 *      timeout = next_pts - mstime
 *      --> read incoming FrameFifo with timeout
 *      --> if frame, call AbstractFileStream->pushFrame(frame)
 *          => AbstractFileStream caches the frame
 * 
 *      (refresh mstime)
 * 
 * 
 * 
 * @ingroup file_tag
 */
class AbstractFileStream {                  // <pyapi>
  
public:                                     // <pyapi>
    /** Default constructor
    * 
    */
    AbstractFileStream();                   // <pyapi>
    virtual ~AbstractFileStream();          // <pyapi>

protected:
    long int    reftime;                ///< Relation between the stream time and wallclock time.  See \ref timing
    long int    target_mstimestamp_;    ///< Where the stream would like to be (underscore means stream time)
    long int    frame_mstimestamp_;     ///< Timestamp of previous frame sent, -1 means there was no previous frame (underscore means stream time)
    AbstractFileState   state;          ///< Decribes the FileStream state: errors, stopped, playing, etc.
    Frame       *next_frame;            ///< Pointer to the next frame about to be presented
    BasicFrame  out_frame;              ///< BasicFrame payload that's passed on to the filterchain
    
public: // getters
    AbstractFileState   getState() {return this->state;}
    
public:
    void setRefMstime(long int ms_streamtime_);  ///< Creates a correspondence with the current wallclock time and a desider stream time, by calculating FileStream::reftime.  See \ref timing
    void play();                                 ///< Start playing the stream
    void stop();                                 ///< Stop playing the strem
    long int update(long int mstimestamp);       ///< Tries to achieve mstimestamp: calculates FileStream::target_mstimestamp_ and calls pullNextFrame.  Returns the timeout for the next frame
    
    virtual void seek(long int ms_streamtime_);  ///< Seek to a desider stream time
    virtual long int pullNextFrame();            ///< Tries to achieve FileStream::target_mstimestamp_ . Sends frames whose timestamps are less than that to the filterchain (e.g. to FileContext::framefilter).  Returns timeout to the next frame.
};                                           // <pyapi>


/** This class uses AbstractFileStream(s)
 * 
 * A minimal example / model class
 * 
 * See also \ref timing.
 * 
 * @ingroup file_tag
 */
class AbstractFileThread : public Thread {              // <pyapi>
    
public:                                                 // <pyapi>
    /** Default constructor
    * 
    * @param name          Thread name
    * 
    */
    AbstractFileThread(const char* name, FrameFifoContext fifo_ctx=FrameFifoContext(10));   // <pyapi>
    virtual ~AbstractFileThread();                                                          // <pyapi>

protected: // frame input
    FrameFifo               infifo;           ///< Incoming frames are read from here
    FifoFrameFilter         infilter;         ///< Write incoming frames here

protected:
    virtual void preRun();
    virtual void postRun();
    virtual void run();
};                                                      // <pyapi>





#endif
