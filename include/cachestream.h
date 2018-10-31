#ifndef cachestream_HEADER_GUARD
#define cachestream_HEADER_GUARD
/*
 * cachestream.h :
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
 *  @file    cachestream.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.8.0 
 *  
 *  @brief
 */

#include "framefifo.h"
#include "filestream.h"


/*   02 03 04 05 06 07 08 09 10 11 12 13
 *   a  a  b  A  a  b  c  C  c  c  B  A
 * 
 *   seek to 10 :  <---- .. key frame for C, A found, but not for B .. if this is from blocks, ok 
 *
 *   better idea: start iterating from the very beginning, save (overwrite) frame pointers during progress 
 * 
 *   Block request:
 *   when to request more blocks / frames ? .. when close enough to maxtime
 * 
 * 
 *   1   2    3
 *       2    3    4
 * 
 */


struct FrameCacheContext { // nothing here yet ..
    int nada;
};


/** FrameCache works like FrameFifo, but frames are not pre-reserved.  They are reserved / released on-the-spot
 * 
 */

class FrameCache {
    
public:
    FrameCache(const char *name, FrameCacheContext ctx =FrameCacheContext());
    ~FrameCache();
    ban_copy_ctor(FrameCache);
    ban_copy_asm(FrameCache);
    
protected:
    std::string       name;
    FrameCacheContext ctx;           ///< Parameters defining the cache
    long int          mintime_;      ///< smallest frame timestamp (frametime)
    long int          maxtime_;      ///< biggest frame timestamp (frametime)
    bool has_delta_frames;           ///< Does the cache have streams with key-frame, delta-frame sequences
  
protected:
    Cache cache;              ///< The queue
    Cache::iterator state;    ///< The state of the queue
  
protected: // mutex synchro
    std::mutex mutex;                         ///< The Lock
    std::condition_variable condition;        ///< The Event/Flag
    std::condition_variable ready_condition;  ///< The Event/Flag for FrameFifo::ready_mutex
  
public: // getters
    long int getMinTime_() {return this->mintime_;}
    long int getMaxTime_() {return this->maxtime_;}
    bool hasDeltaFrames()  {return this->has_delta_frames;}
      
      
public:
    virtual bool writeCopy(Frame* f, bool wait=false);     ///< Take a frame "ftmp" from the stack, copy contents of "f" into "ftmp" and insert "ftmp" into the beginning of the fifo (i.e. perform "copy-on-insert").  The size of "ftmp" is also checked and set to target_size, if necessary.  If wait is set to true, will wait until there are frames available in the stack.
    // virtual Frame* read(unsigned short int mstimeout=0);   ///< Pop a frame from the end of the fifo and return the frame to the reservoir stack
    void clear();          ///< Clear the cache
    void dump();           ///< Dump frames in the cache
    bool isEmpty();        ///< Cache empty or not 
    int seek(long int ms_streamtime_);  ///< Seek to a desider stream time.  -1 = no frames at left, 1 = no frames at right, 0 = ok
    Frame *pullNextFrame();            ///< Get the next frame.  Returns NULL if no frame was available
};



/** Passes frames to a FrameCache
 * 
 * Typically, the terminal point for the frame filter chain, so there is no next filter = NULL.
 * 
 * @ingroup filters_tag
 * @ingroup queues_tag
 */
class CacheFrameFilter : public FrameFilter {                                            // <pyapi>
  
public:                                                                                 // <pyapi>
  /** Default constructor
   * 
   * @param name       Name
   * @param framecache The FrameCache where the frames are being written
   */
  CacheFrameFilter(const char* name, FrameCache* framecache); ///< Default constructor     // <pyapi>
  
protected:
  FrameCache* framecache;
  
protected:
  void go(Frame* frame);
};                                                                                      // <pyapi>



/** Caches (a large amount of) frames and pushes them forward at a rate corresponding to play speed
 * 
 *  - CacheFileStream admins FrameFilter and FrameCache (that is protected with a mutex)
 *  - CacheFileStream only observes (does not manipulate) FrameCache, that is being filled and run by the Filesystem thread
 *  - FrameCache has its own rules how to behave when start / end frames are reserved, how the frames are updated, etc.
 *  - FileCacheThread "runs" and times the CacheFileStream.
 *  - Filesystem thread gets the FrameFilter from CacheFileThread that gets it from CacheFileStream
 *  - Filesytem thread does the cam id => slot mapping
 * 
 * 
 *  CacheStream
 *      CacheFrameFilter (pyapi) --> FrameCache
 *      getFrameFilter : returns CacheFrameFilter           
 *  
 *  FileCacheThread (pyapi)
 *      uses CacheStream
 *      getFrameFilter : uses CacheStream::getFrameFilter
 *      Similar to LiveThread, give output contexes (certain slot to certain framefilter)
 * 
 *  Filesystem thread writes to FileCacheThread::getFrameFilter
 * 
 * 
 */
class CacheStream : public AbstractFileStream {

public:
    CacheStream(const char *name, FrameCacheContext cache_ctx = FrameCacheContext());
    virtual ~CacheStream(); ///< Default virtual destructor
        
protected:
    FrameCache          cache;        
    CacheFrameFilter    infilter;
    AbstractFileState   state;
    long int    reftime;                ///< Relation between the stream time and wallclock time.  See \ref timing
    long int    target_mstimestamp_;    ///< Where the stream would like to be (underscore means stream time)
    long int    frame_mstimestamp_;     ///< Timestamp of previous frame sent, -1 means there was no previous frame (underscore means stream time)
    
public:
    std::map<SlotNumber, FrameFilter*>  slots_;     ///< Manipulated by the thread that's using CacheStream
    
public: // getters
    long int getMinTime_() {return cache.getMinTime_();}
    long int getMaxTime_() {return cache.getMaxTime_();}
    CacheFrameFilter &getFrameFilter() {return this->infilter;}
    void dumpCache() {cache.dump();}
    
public: // virtual reimplemented
    void seek(long int ms_streamtime_);  ///< Set the state of CacheStream::cache to frame corresponding to this time
    
    /** Pulls the next frame
     * - Transmits this->next_frame
     * - Gets the next frame in going towards the target_mstimestamps_
     * - .. and puts it into this->next_frame
     * - Returns the timeout, i.e. the difference between target_mstimestamp_ and this->next_frame.mstimestamp
     * - If timeout is = 0, calls itself recursively
     */
    long int pullNextFrame();
    long int update(long int mstimestamp);       ///< Calculates FileStream::target_mstimestamp_ and calls pullNextFrame.  Returns the timeout for the next frame
    
    
};


struct FileCacheSignalContext {                                                                       // <pyapi>
  /** Default constructor */
  FileCacheSignalContext(SlotNumber slot, FrameFilter* framefilter) :                                 // <pyapi>
  slot(slot), framefilter(framefilter)                                                                // <pyapi>
  {}                                                                                                  // <pyapi>
  /** Dummy constructor : remember to set member values by hand */
  FileCacheSignalContext() :                                                                          // <pyapi>
  slot(0), framefilter(NULL)                                                                          // <pyapi>  
  {}                                                                                                  // <pyapi>
  SlotNumber         slot;              ///< A unique stream slot that identifies this stream         // <pyapi>
  FrameFilter*       framefilter;       ///< The frames are feeded into this FrameFilter              // <pyapi>
};                                                                                                    // <pyapi>


class FileCacheThread : public AbstractFileThread {                           // <pyapi>
    
public:                                                                     // <pyapi>
    FileCacheThread(const char *name);                                      // <pyapi>
    virtual ~FileCacheThread();                                             // <pyapi>
        
protected:
    CacheStream           stream;
    
protected: // Thread member redefinitions
    std::deque<FileCacheSignalContext> signal_fifo;   ///< Redefinition of signal fifo.
  
public: // redefined virtual functions
    void run();
    void preRun();
    void postRun();
    void sendSignal(FileCacheSignalContext signal_ctx);
  
protected:
    void handleSignals();

private: // internal
    void registerStream   (FileCacheSignalContext &ctx);
    void deregisterStream (FileCacheSignalContext &ctx);
  
public: // API                                              // <pyapi>
    void registerStreamCall   (FileCacheSignalContext &ctx); ///< API method: registers a stream                                // <pyapi> 
    void deregisterStreamCall (FileCacheSignalContext &ctx); ///< API method: de-registers a stream                             // <pyapi>
    const CacheFrameFilter &getFrameFilter();                                            // <pyapi>
    void requestStopCall();                                                             // <pyapi>
    // TODO: stop, play, seek commands
};                                                                                      // <pyapi>




#endif
