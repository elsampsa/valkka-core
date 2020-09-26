#ifndef avfilethread_HEADER_GUARD 
#define avfilethread_HEADER_GUARD

/*
 * avfilethread.h : A Thread handling files and sending frames to fifo
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

// TODO: implement this as a subclass of AbstractFileThread
// TODO: rename FileThread to AVFileThread

/** 
 *  @file    avfilethread.h
 *  @author  Sampsa Riikonen
 *  @date    2018
 *  @version 1.0.0 
 *  
 *  @brief A Thread handling files and sending frames to fifo
 */ 


#include "frame.h"
#include "thread.h"
#include "framefilter.h"
#include "logging.h"
#include "tools.h"

/** Describes the state of a FileStream 
 *
 * @ingroup file_tag
 */
enum class FileState {                      // <pyapi>
  none,                                     // <pyapi>
  error,                                    // <pyapi>
  seek, // in the middle of a seek          // <pyapi>
  stop, // stream stopped                   // <pyapi>
  play  // stream is playing                // <pyapi> 
};                                          // <pyapi>


/** This class descibes the origin and state of a FileStream.
 * 
 * It's also used by FileSignalContext to carry signal information.
 * 
 * Two different constructors are provided, the one without arguments can be used to create "dummy" objects.
 * 
 * @ingroup file_tag
 */
struct FileContext {                                                                               // <pyapi>
  FileContext(std::string filename, SlotNumber slot, FrameFilter* framefilter, long int st=0) :    // <pyapi>
  filename(filename), slot(slot), framefilter(framefilter), seektime_(st),                         // <pyapi>
  duration(0), mstimestamp(0), status(FileState::none)                                             // <pyapi>
  {}                              ///< Default constructor                                         // <pyapi>
  FileContext() : filename(""), slot(0), framefilter(NULL), seektime_(0),                          // <pyapi>
  duration(0), mstimestamp(0), status(FileState::none)                                             // <pyapi>
  {}                              ///< Dummy constructor.  Set values by manipulating members      // <pyapi> 
  std::string    filename;        ///< incoming: the filename                                      // <pyapi>
  SlotNumber     slot;            ///< incoming: a unique stream slot that identifies this stream  // <pyapi>
  FrameFilter*   framefilter;     ///< incoming: the frames are feeded into this FrameFilter       // <pyapi>
  long int       seektime_;       ///< incoming: used by signal seek_stream                        // <pyapi>
  long int       duration;        ///< outgoing: duration of the stream                            // <pyapi>
  long int       mstimestamp;     ///< outgoing: current position of the stream (stream time)      // <pyapi>
  FileState      status;          ///< outgoing: status of the file                                // <pyapi>
};                                                                                                 // <pyapi>


// TODO: still ugly memory leaks with the bitstream filter.  Use this class to test things.
class TestFileStream {
  public:
  /** Default constructor
   * 
   * @param ctx : A FileContext instance describing the stream
   */
  TestFileStream(const char* filename);
  ~TestFileStream(); ///< Default destructor
  
public:
  AVFormatContext     *input_context;
 
public:
  AVPacket *avpkt;                 ///< Data for the next frame in ffmpeg AVPacket format
  AVBitStreamFilterContext *annexb;
  void     pull();                   
};


/** This class in analogous to the Connection class in live streams.  Instances of this class are placed in slots in FileThread.
 * 
 * This class "keeps the books" for each file stream, in particular:
 *
 * - Desider target time (FileStream::target_mstimestamp_)
 * - Timestamp of the previous frame (FileStream::frame_mstimestamp_)
 * - State of the stream (FileStream::state)
 * - FFmpeg stream handles
 * - etc.
 * 
 * In variable names, underscore means stream time.  See \ref timing
 * 
 * @ingroup file_tag
 */
class FileStream {
  
public:
  /** Default constructor
   * 
   * @param ctx : A FileContext instance describing the stream
   */
  FileStream(FileContext &ctx);
  ~FileStream(); ///< Default destructor
  
public:
  FileContext         &ctx;           ///< FileContext describing this stream
  AVFormatContext     *input_context;
  std::vector<AVBitStreamFilterContext*> filters;
  
public:
  SetupFrame  setupframe;             ///< Setup frame written to the filterchain
  BasicFrame  out_frame;              ///< This frame is written to the filterchain (i.e. to FileStream::ctx and there to FileContext::framefilter) 
  long int    duration;               ///< Duration of the stream
  long int    reftime;                ///< Relation between the stream time and wallclock time.  See \ref timing
  long int    target_mstimestamp_;    ///< Where the stream would like to be (underscore means stream time)
  long int    frame_mstimestamp_;     ///< Timestamp of previous frame sent, -1 means there was no previous frame (underscore means stream time)
  FileState   state;                  ///< Decribes the FileStream state: errors, stopped, playing, etc.
  AVPacket    *avpkt;                 ///< Data for the next frame in ffmpeg AVPacket format
  
public: // getters
  SlotNumber getSlot() {return ctx.slot;}
  
public:
  void setRefMstime(long int ms_streamtime_);  ///< Creates a correspondence with the current wallclock time and a desider stream time, by calculating FileStream::reftime.  See \ref timing
  void seek(long int ms_streamtime_);          ///< Seek to a desider stream time
  void play();                                 ///< Start playing the stream
  void stop();                                 ///< Stop playing the strem
  long int update(long int mstimestamp);       ///< Tries to achieve mstimestamp: calculates FileStream::target_mstimestamp_ and calls pullNextFrame.  Returns the timeout for the next frame
  long int pullNextFrame();                    ///< Tries to achieve FileStream::target_mstimestamp_ . Sends frames whose timestamps are less than that to the filterchain (e.g. to FileContext::framefilter).  Returns timeout to the next frame.
};


/** Characteristic signals for the FileThread.
* 
* These signals map directly into methods with the same names
* 
*/
enum class FileSignal {
  none,
  exit,                 
  open_stream,
  close_stream,
  seek_stream,
  play_stream,
  stop_stream,
  get_state            ///< query information about the stream
};


/** Identifies the information the signals FileSignal carry.  Encapsulates a FileContext instance.
*
*/
struct FileSignalContext {
  FileSignal  signal;
  FileContext *file_context; ///< pointer, cause we have return values
};


/** This class in analogous to LiveThread, but it handles files instead of live streams.
 * 
 * FileThread's execution loop works roughly as follows:
 * 
 * - Each active slot has a FileStream instance
 * - For each FileStream, FileStream::update is called with the current wallclock time
 * - FileStream::update returns a timeout to the next frame in the file - in the case of seek, this is zero untill the target time is met
 * 
 * See also \ref timing.
 * 
 * @ingroup file_tag
 */
class FileThread : public Thread { // <pyapi>

public:                                                // <pyapi>
  /** Default constructor
   * 
   * @param name          Thread name
   */
  FileThread(const char* name, FrameFifoContext fifo_ctx=FrameFifoContext());        // <pyapi>
  /** Default destructor */
  ~FileThread();                                       // <pyapi>
  
protected: // frame input // TODO: implement writing to files with FileThread
  FrameFifo         infifo;     ///< A FrameFifo for incoming frames
  FifoFrameFilter   infilter;   ///< A FrameFilter for writing incoming frames
  
protected: // redefinitions
  std::deque<FileSignalContext> signal_fifo;    ///< Redefinition of signal fifo (Thread::signal_fifo is now hidden from usage) 
  
protected:
  std::vector<FileStream*> slots_;          ///< Slots: a vector of FileStream instances
  // TODO: all slots should be done in the future with std::map<SlotNumber,FileStream*>
  std::list<SlotNumber>   active_slots;     ///< Slots that are activated 
  bool loop;                                ///< Controls the execution of the main loop
  // std::list<FileStream*>  streamlist;       // TODO: a better event loop: FileStream s that have frames to be presented are queued here
  
/* // some misc. ideas ..
protected:
  int                       count_streams_seeking; ///< number of stream seeking at the moment
  std::condition_variable   seek_condition;        ///< notified when all streams have stopped seeking 
*/
  

public: // redefined virtual functions
  void run();
  void preRun();
  void postRun();
  void sendSignal(FileSignalContext signal_ctx);         
  void sendSignalAndWait(FileSignalContext signal_ctx);  
  
protected:
  void handleSignals();
  
private: // internal
  int  safeGetSlot          (SlotNumber slot, FileStream*& stream);
  void openFileStream       (FileContext &file_ctx);
  void closeFileStream      (FileContext &file_ctx);
  void seekFileStream       (FileContext &file_ctx);
  void playFileStream       (FileContext &file_ctx);
  void stopFileStream       (FileContext &file_ctx);
  
public: // *** C & Python API *** .. these routines go through the condvar/mutex locking                                     // <pyapi>
  void closeFileStreamCall      (FileContext &file_ctx); ///< API method: registers a stream                                // <pyapi> 
  void openFileStreamCall       (FileContext &file_ctx); ///< API method: de-registers a stream                             // <pyapi>
  void seekFileStreamCall       (FileContext &file_ctx); ///< API method: seek to a certain point                           // <pyapi>
  void playFileStreamCall       (FileContext &file_ctx); ///< API method: starts playing the stream and feeding frames      // <pyapi>
  void stopFileStreamCall       (FileContext &file_ctx); ///< API method: stops playing the stream and feeding frames       // <pyapi>
  void requestStopCall();                                ///< API method: Like Thread::stopCall() but does not block        // <pyapi>
  FifoFrameFilter &getFrameFilter();                     ///< API method: get filter for sending frames with live555        // <pyapi>
};                                                                                                                          // <pyapi>

#endif
