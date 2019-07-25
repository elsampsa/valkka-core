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
 *  @version 0.13.0 
 *  
 *  @brief
 */

#include "framefifo.h"
#include "filestream.h"
#include "Python.h"


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



struct FileStreamContext {                                          // <pyapi>
    FileStreamContext(SlotNumber slot, FrameFilter *framefilter) :  // <pyapi>
    slot(slot), framefilter(framefilter) {}                         // <pyapi>
    FileStreamContext() {}                                          // <pyapi>
    SlotNumber      slot;                                           // <pyapi>
    FrameFilter     *framefilter;                                   // <pyapi>
};                                                                  // <pyapi>


/** Signal information for FileCacheThread
 * 
 */
struct FileCacheSignalPars {                                
    ///< Identifies the stream                             
    FileStreamContext   file_stream_ctx;                    
    ///< Timestamp for the seek signal
    long int mstimestamp;                                   
    ///< Seek: use existing frames for seek or clear the state
    bool clear;
}; 


/** Signals for FileCacheThread
 * 
 */
enum class FileCacheSignal {
    none,
    exit,
    register_stream,
    deregister_stream,
    play_streams,
    stop_streams,
    seek_streams,
    clear_streams,
    report_cache
};


/** Encapsulate data sent to FileCacheThread
 * 
 * 
 */
struct FileCacheSignalContext {
    FileCacheSignal         signal;
    FileCacheSignalPars     pars;
};



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
    FrameCacheContext ctx;              ///< Parameters defining the cache
    long int          mintime_;         ///< smallest frame timestamp (frametime)
    long int          maxtime_;         ///< biggest frame timestamp (frametime)
    bool              has_delta_frames; ///< Does the cache have streams with key-frame, delta-frame sequences
  
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
      
protected:
    void dump_();
    void clear_();
      
public:
    virtual bool writeCopy(Frame* f, bool wait=false);     ///< Take a frame "ftmp" from the stack, copy contents of "f" into "ftmp" and insert "ftmp" into the beginning of the fifo (i.e. perform "copy-on-insert").  The size of "ftmp" is also checked and set to target_size, if necessary.  If wait is set to true, will wait until there are frames available in the stack.
    // virtual Frame* read(unsigned short int mstimeout=0);   ///< Pop a frame from the end of the fifo and return the frame to the reservoir stack
    void clear();          ///< Clear the cache
    void dump();           ///< Dump frames in the cache
    bool isEmpty();        ///< Cache empty or not 
    int seek(long int ms_streamtime_);  ///< Seek to a desider stream time.  -1 = no frames at left, 1 = no frames at right, 0 = ok
    int keySeek(long int ms_streamtime_);  ///< Seek to a desider stream time.  -1 = no frames at left, 1 = no frames at right, 0 = ok.  From the nearest key-frame to the target frame.
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


/** Thread that caches frames and streams them into output at play speed
 * 
 * - Has two FrameCaches: other one is used to stream, while other one is caching the incoming frames
 * - This framefilter chain is typically driven by another thread (ValkkaFSReaderThread)
 * 
 * \verbatim
 * 
 *                                                       +----> FrameCache
 *                                                       |
 *                           +----> SwitchFrameFilter ---+                This part of the framefilter chain is typically 
 *                           |                           |                driven by another thread (ValkkaFSReaderThread)
 *                           |                           +----> FrameCache
 *  ---> ForkFrameFilter ----+
 *                           |
 *                           |
 *                           +----> ClassFrameFilter (MarkerFrame) ---> FileCacheThread event loop
 * 
 * \endverbatim
 * 
 * 
 * 
 */
class FileCacheThread : public AbstractFileThread {                                     // <pyapi>
    
public:                                                                                 // <pyapi>
    FileCacheThread(const char *name);                                                  // <pyapi>
    virtual ~FileCacheThread();                                                         // <pyapi>
        
protected: // internal framefilter chain
    ForkFrameFilter     fork;                ///< Write incoming frames here
    TypeFrameFilter     typefilter;
    SwitchFrameFilter   switchfilter;
    CacheFrameFilter    cache_filter_1, cache_filter_2; 
    
protected:
    FrameCache  frame_cache_1;
    FrameCache  frame_cache_2;
    FrameCache  *play_cache;    ///< Points to the current play cache (default frame_cache_2)
    FrameCache  *tmp_cache;     ///< Points to current cache receiving frames
    void        (*callback)(long int mstimestamp);
    PyObject    *pyfunc;        ///< Python callback that emits current time
    PyObject    *pyfunc2;       ///< Python callback that emits current loaded time limits
    long int    target_mstimestamp_;    ///< We should be at this time instant (streamtime)
    Frame       *next;
    long int    reftime;                ///< walltime = frametime_ + reftime
    long int    walltime;
    AbstractFileState state;
    SetupFrame  state_setupframe; ///< SetupFrame for sending the stream state (seek, play, etc.)
    
protected: // Thread member redefinitions
    std::deque<FileCacheSignalContext> signal_fifo;   ///< Redefinition of signal fifo.
    std::vector<FrameFilter*>          slots_;        ///< Slot number => output framefilter mapping
    // std::vector<SetupFrame*>           setup_frames;  ///< Slot number => SetupFrame mapping.  Book-keeping of SetupFrames
    std::vector<std::vector<SetupFrame*>>   setup_frames; ///< Slot number, subsession_index => SetupFrame mapping.  Book-keeping of SetupFrames
    
    
public: // redefined virtual functions
    void run();
    void preRun();
    void postRun();
    void sendSignal(FileCacheSignalContext signal_ctx);
  
public: // internal
    void switchCache();
    void dumpPlayCache();
    void dumpTmpCache();
    void sendSetupFrames(SetupFrame *f);                                ///< Sends SetupFrame s to all active slots
    void stopStreams(bool send_state = true);
    void playStreams(bool send_state = true);
    void setRefTimeAndStop(bool send_state = true);                     ///< Set reference time and set state to stop
    void seekStreams(long int mstimestamp, bool clear, bool send_state = true);     ///< Sets target time.  Sets FileCacheThread::next = NULL
    
private: // internal
    void handleSignal(FileCacheSignalContext &signal_ctx);
    void handleSignals();
    int  safeGetSlot(SlotNumber slot, FrameFilter*& ff);
    void registerStream   (FileStreamContext &ctx);
    void deregisterStream (FileStreamContext &ctx);
  
public: // API must be called before thread start
    void setCallback(void func(long int));
    
public: // API // <pyapi>
    /** Define callback 
     * - Give python function that is called frequently as the play time changes
     * - Function should have a single argument (int)
     */
    void setPyCallback(PyObject* pobj);                     // <pyapi>
    
    /** Define callback
     * - Give python function that is called when new frames are cached
     * - Function should have a single argument (tuple)
     * - first element is the minimum timestamp
     * - second element is the maximum timestamp
     */
    void setPyCallback2(PyObject* pobj);                    // <pyapi>
    
    /** Pass frames downstream
     * - Define filter where frames are passed downstream
     * - map from ValkkaFS id number to slot number
     */
    void registerStreamCall   (FileStreamContext &ctx);     // <pyapi> 
    void deregisterStreamCall (FileStreamContext &ctx);     // <pyapi>
    
    /** Framefilter for writing frames to FileCacheThread */
    FrameFilter &getFrameFilter();                          // <pyapi>
    void requestStopCall();                                 // <pyapi>
    void dumpCache();                                       // <pyapi>
    void stopStreamsCall();                                 // <pyapi>
    void playStreamsCall();                                 // <pyapi>
    
    /** Seek
     * - Seek to a certain millisecond timestamp
     * @param mstimestamp The unix millisecond timestamp
     * @param clear       False (default) = Seek within frames already in the cache.  Don't clear internal reference time.  True = We're expecting a burst of new frames into the cache (new transmission).  Clear internal reference time.
     *  
     * 
     */
    void seekStreamsCall(long int mstimestamp, bool clear = false); // <pyapi>
};                                                                  // <pyapi>




#endif
