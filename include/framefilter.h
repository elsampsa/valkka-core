#ifndef framefilter_HEADER_GUARD
#define framefilter_HEADER_GUARD
/*
 * framefilter.h : Definition of FrameFilter and derived classes for various purposes
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
 *  @file    framefilter.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.8.0 
 *  
 *  @brief   Definition of FrameFilter and derived classes for various purposes
 */ 


#include "frame.h"
#include "framefifo.h"


/** The mother class of all frame filters!  
 * FrameFilters are used to create "filter chains".  These chains can be used to manipulate the frames, feed them to fifo's, copy them, etc.
 * 
 * @ingroup filters_tag
 */
class FrameFilter {                                                               // <pyapi>
  
public:                                                                           // <pyapi>
  /** Default constructor
   * 
   * @param name  Name of the filter
   * @param next  Next FrameFilter instance in the filter chain
   * 
   */
  FrameFilter(const char* name, FrameFilter* next=NULL); // don't include into the python api (this class is abstract)
  virtual ~FrameFilter();                                ///< Virtual destructor // <pyapi>
  
protected:
  std::string  name;
  FrameFilter* next; ///< The next frame filter in the chain to be applied
  
protected:                                                                      // <pyapi>
  // does the filtering 
  virtual void go(Frame* frame) = 0; ///< Does the actual filtering/modification to the Frame.  Define in subclass
  
public: // API
  /** Calls this->go(Frame* frame) and then calls the this->next->run(Frame* frame) (if this->next != NULL)
   */
  virtual void run(Frame* frame);
};                                                                              // <pyapi>


/** A "hello world" demo class: prints its own name if verbose is set to true.
 * @ingroup filters_tag
 */
class DummyFrameFilter : public FrameFilter {                                    // <pyapi>
  
public:                                                                          // <pyapi>
  DummyFrameFilter(const char* name, bool verbose=true, FrameFilter* next=NULL); // <pyapi>
  
protected:
  bool verbose;
  
protected:
  void go(Frame* frame);
  
};                                                                               // <pyapi>


/** Dump the beginning of Frame's payload into stdout
 * @ingroup filters_tag
 */
class InfoFrameFilter : public FrameFilter {                                    // <pyapi>
  
public:                                                                         // <pyapi>
  InfoFrameFilter(const char* name, FrameFilter* next=NULL);                    // <pyapi>
    
protected:
  void go(Frame* frame);
};                                                                              // <pyapi>


/** Dump the beginning of Frame's payload into stdout in a one-liner
 * @ingroup filters_tag
 */
class BriefInfoFrameFilter : public FrameFilter {                               // <pyapi>
  
public:                                                                         // <pyapi>
  BriefInfoFrameFilter(const char* name, FrameFilter* next=NULL);               // <pyapi>
    
protected:
  void go(Frame* frame);
};                                                                              // <pyapi>


/** Replicates frame flow to two filters
 * Use this frame filter to create frame filter tree structures
 * @ingroup filters_tag
 */
class ForkFrameFilter : public FrameFilter {                                    // <pyapi>
  
public:                                                                         // <pyapi>
  /** @copydoc FrameFilter::FrameFilter
   * 
   *  @param next2 Yet another next FrameFilter instance to be applied in the chain
   * 
   */
  ForkFrameFilter(const char* name, FrameFilter* next=NULL, FrameFilter* next2=NULL); // <pyapi>

protected:
  FrameFilter* next2;

protected:
  void go(Frame* frame);
  
public:
  void run(Frame* frame); 
};                                                                            // <pyapi>


/** Replicates frame flow to three filters
 * Use this frame filter to create frame filter tree structures
 * @ingroup filters_tag
 */
class ForkFrameFilter3 : public FrameFilter {                                 // <pyapi>
  
public:                                                                       // <pyapi>
  /** @copydoc FrameFilter::FrameFilter
   * 
   *  @param next2 Yet another next FrameFilter instance to be applied in the chain
   *  @param next3 Still yet another next FrameFilter instance to be applied in the chain
   * 
   */
  ForkFrameFilter3(const char* name, FrameFilter* next=NULL, FrameFilter* next2=NULL, FrameFilter* next3=NULL); // <pyapi>

protected:
  FrameFilter* next2;
  FrameFilter* next3;

protected:
  void go(Frame* frame);
  
public:
  void run(Frame* frame); 
};                                                                          // <pyapi>


/** Replicates frame flow to arbitrary number of outputs
 * 
 * - Terminals are added after the instance has been created
 *
 * @ingroup filters_tag 
 */
class ForkFrameFilterN : public FrameFilter {                             // <pyapi>
  
public:                                                                   // <pyapi>
  /** Default ctor
   * 
   * @param name  Name identifying this FrameFilter
   * 
   */
  ForkFrameFilterN(const char* name);                                     // <pyapi>
  /** Default virtual dtor
   */
  virtual ~ForkFrameFilterN();                                            // <pyapi>
  
protected:
  std::mutex  mutex;
  std::map<std::string,FrameFilter*> framefilters;  ///< nametag to connecting FrameFilter mapping
  
protected:
  void go(Frame* frame);
  
public:
  void run(Frame* frame); ///< called by other FrameFilter(s)
  
public:                                                                   // <pyapi>
  /** Connect a new terminal FrameFilter.  Tag the connection with a name.
   * 
   * @param tag     Nametag for this connection
   * @param filter  FrameFilter for the connection
   * 
   */
  bool connect    (const char* tag, FrameFilter* filter);                 // <pyapi>
  /** Disconnect a connection tagged with a name
   * 
   * @param tag     Nametag for this connection
   * 
   */
  bool disconnect (const char* tag);                                      // <pyapi>
};                                                                         // <pyapi>




/** Sets the frame slot value
 * @ingroup filters_tag
 */
class SlotFrameFilter : public FrameFilter {                                // <pyapi>
  
public:                                                                     // <pyapi>
  SlotFrameFilter(const char* name, SlotNumber n_slot, FrameFilter* next=NULL);  // <pyapi>
    
protected:
  unsigned n_slot;
  
protected:
  void go(Frame* frame);
  
};                                                                          // <pyapi>


/** Dumps each received packet to a file: use with care!  For debugging purposes only.
 * @ingroup filters_tag
 */
class DumpFrameFilter : public FrameFilter {                                // <pyapi>
  
public:                                                                     // <pyapi>
  DumpFrameFilter(const char* name, FrameFilter* next=NULL);                // <pyapi>
  
protected:
  int count;
  
protected:
  void go(Frame* frame);
};                                                                          // <pyapi>


/** Counts frames passed through this filter
 * @ingroup filters_tag
 */

class CountFrameFilter : public FrameFilter {                             // <pyapi>
  
public:                                                                   // <pyapi>
  CountFrameFilter(const char* name, FrameFilter* next=NULL);             // <pyapi>
  
protected:
  int count;
  
protected:
  void go(Frame* frame);
};                                                                        // <pyapi> 



/** Corrects erroneous timestamps (while preserving timestamp distances).
 * @ingroup filters_tag
 */
class TimestampFrameFilter : public FrameFilter {                         // <pyapi>
  
public:                                                                   // <pyapi>
  TimestampFrameFilter(const char* name, FrameFilter* next=NULL, long int msdiff_max=TIMESTAMP_CORRECT_TRESHOLD);  // <pyapi>
    
protected:
  long int mstime_delta;
  long int msdiff_max;
  
protected: 
  void go(Frame* frame);
};                                                                        // <pyapi>



/** Corrects erroneous timestamps (while preserving timestamp distances).  Reset correction every 10 minutes.
 * @ingroup filters_tag
 */
class TimestampFrameFilter2 : public FrameFilter {                        // <pyapi>
  
public:                                                                   // <pyapi>
  TimestampFrameFilter2(const char* name, FrameFilter* next=NULL, long int msdiff_max=TIMESTAMP_CORRECT_TRESHOLD);  // <pyapi>
    
protected:
  long int mstime_delta;
  long int msdiff_max;
  long int savedtimestamp;
  
protected: 
  void go(Frame* frame);
};                                                                        // <pyapi>


/** Substitute timestamps with the time they arrive to the client.
 * @ingroup filters_tag
 */
class DummyTimestampFrameFilter : public FrameFilter {                    // <pyapi>
  
public:                                                                   // <pyapi>
  DummyTimestampFrameFilter(const char* name, FrameFilter* next=NULL);    // <pyapi>

protected: 
  void go(Frame* frame);
};                                                                        // <pyapi>


/** For H264, some cameras don't send sps and pps packets again before every keyframe.  In that case, this filter sends sps and pps before each keyframe.
 * 
 * WARNING: not ready .. in the TODO list
 * 
 * @ingroup filters_tag
 */
class RepeatH264ParsFrameFilter : public FrameFilter {                  // <pyapi>
  
public:                                                                 // <pyapi>
  RepeatH264ParsFrameFilter(const char* name, FrameFilter* next=NULL);  // <pyapi>
    
protected:
  Frame sps_frame, pps_frame;
  
protected:
  void go(Frame* frame);
  
};                                                                      // <pyapi>


/** When turned on, passes frames.  When turned off, frames are not passed.  
 * 
 * - Configuration frames (FrameType::setup) are passed, even if the gate is unSet
 * - Passing of configuration frames can be turned off (when gate is unSet) by calling noConfigFrames()
 * - Mutex-protected (calls to GateFrameFilter::set and GateFrameFilter::unSet happen during streaming)
 * 
 * @ingroup filters_tag
 */
class GateFrameFilter : public FrameFilter {                          // <pyapi>
  
public:                                                               // <pyapi>
  GateFrameFilter(const char* name, FrameFilter* next=NULL);          // <pyapi>
  
protected:
  bool        on;
  bool        config_frames;
  std::mutex  mutex;
  
protected: 
  void run(Frame* frame);
  void go(Frame* frame);
  
public:                     // <pyapi>
  void set();               // <pyapi>
  void unSet();             // <pyapi>
  void passConfigFrames();  // <pyapi>
  void noConfigFrames();    // <pyapi>
};                                                                   // <pyapi>


/** Caches SetupFrame s
 * 
 * Like GateFrameFilter, but caches SetupFrame s and re-emits them always when the gate is activated
 * 
 */
class CachingGateFrameFilter : public FrameFilter {                   // <pyapi> 
 
public:                                                               // <pyapi>
  CachingGateFrameFilter(const char* name, FrameFilter* next=NULL);   // <pyapi>
  
protected:
  bool        on;
  std::mutex  mutex;
  SetupFrame  setupframe;
  bool        got_setup;
  
protected: 
  void run(Frame* frame);
  void go(Frame* frame);
  
public:                     // <pyapi>
  void set();               // <pyapi>
  void unSet();             // <pyapi>
};                                                                   // <pyapi>



/** Changes the slot number of the Frame
 *
 * Mutex-protected (calls to SetSlotFrameFilter::setSlot happen during streaming)
 * 
 * @ingroup filters_tag 
 */
class SetSlotFrameFilter : public FrameFilter {                     // <pyapi>
  
public: // <pyapi>
  SetSlotFrameFilter(const char* name, FrameFilter* next=NULL);     // <pyapi>

protected:
  SlotNumber n_slot;
  std::mutex mutex;
  
protected: 
  void go(Frame* frame);
  
public:                         // <pyapi>
  void setSlot(SlotNumber n=0); // <pyapi>
  
};                                                                  // <pyapi>
  

/** Pass frames, but not all of them - only on regular intervals.  This serves for downshifting the fps rate.
 *
 * Should be used, of course, only for decoded frames..!
 * 
 * @param name          A name identifying the frame filter
 * @param mstimedelta   Time interval in milliseconds
 * @param next          Next filter in chain
 * 
 * @ingroup filters_tag
 */
class TimeIntervalFrameFilter : public FrameFilter {                                      // <pyapi>
  
public:                                                                                   // <pyapi>
  TimeIntervalFrameFilter(const char* name, long int mstimedelta, FrameFilter* next=NULL); // <pyapi>

protected:
  long int mstimedelta; 
  long int prevmstimestamp;
  
protected: 
  void go(Frame* frame);
  
public:
  void run(Frame* frame);
    
};                                                                                        // <pyapi>


/** Passes frames to a FrameFifo
 * 
 * Typically, the terminal point for the frame filter chain, so there is no next filter = NULL.
 * 
 * @ingroup filters_tag
 * @ingroup queues_tag
 */
class FifoFrameFilter : public FrameFilter {                                            // <pyapi>
  
public:                                                                                 // <pyapi>
  /** Default constructor
   * 
   * @param name       Name
   * @param framefifo  The FrameFifo where the frames are being written
   */
  FifoFrameFilter(const char* name, FrameFifo* framefifo); ///< Default constructor     // <pyapi>
  
protected:
  FrameFifo* framefifo;
  
protected:
  void go(Frame* frame);
};                                                                                      // <pyapi>


/** Passes frames to a multiprocessing fifo.
 * 
 * Works like FifoFrameFilter, but blocks if the receiving FrameFifo does not have available frames
 * 
 * @ingroup filters_tag
 * @ingroup queues_tag
 */
class BlockingFifoFrameFilter : public FrameFilter {                                          // <pyapi>
  
public:                                                                                       // <pyapi>
  BlockingFifoFrameFilter(const char* name, FrameFifo* framefifo); ///< Default constructor // <pyapi>
  
protected:
  FrameFifo* framefifo;
  
protected:
  void go(Frame* frame);
};                                                                                              // <pyapi>



/** Interpolate from YUV bitmap to RGB
 * 
 * - Creates internal, outbound frame with reserved space for AVFrame data
 * - AVThread writes a Frame with it's av_frame (AVFrame) instance activated, into this FrameFilter
 * - When first frame arrives here, reserve SwsContext (SwScaleFrameFilter::sws_ctx)
 * - If incoming frame's dimensions change, re-reserve sws_ctx
 * 
 * @ingroup filters_tag
 */
class SwScaleFrameFilter : public FrameFilter {                                                 // <pyapi>
  
public: // <pyapi>
  SwScaleFrameFilter(const char* name, int target_width, int target_height, FrameFilter* next=NULL); ///< Default constructor // <pyapi>
  ~SwScaleFrameFilter(); ///< Default destructor                                              // <pyapi>
  
protected: // initialized at constructor
  int            target_width;       ///< target frame width
  int            target_height;      ///< target frame height
  int            width;
  int            height;
  AVRGBFrame     outputframe;  
  SwsContext    *sws_ctx;        ///< FFmpeg scaling context structure
  
protected:
  void go(Frame* frame);
  
public:
  void run(Frame* frame);
};                                                                                            // <pyapi>
  



#endif
