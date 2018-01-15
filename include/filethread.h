/*
 * filethread.h : A Thread handling files and sending frames to fifo
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
 *  @file    filethread.h
 *  @author  Sampsa Riikonen
 *  @date    2018
 *  @version 0.2
 *  
 *  @brief A Thread handling files and sending frames to fifo
 */ 


#include "frames.h"
#include "threads.h"
#include "logging.h"
#include "tools.h"


enum class FileState {
  none,
  error,
  seek, // in the middle of a seek
  stop, // stream stopped
  play  // stream is playing
};


/** Keeping the books for each stream: the reference time (see \ref timing) and the state of the stream */
class FileStream {
  
public:
  FileStream(std::string filename, SlotNumber slot, FrameFilter& framefilter);
  ~FileStream();
  
public:
  std::string         filename;        ///< FileStream address
  SlotNumber          slot;            ///< FileStream slot number (that identifies the source)
  FrameFilter         &framefilter;    ///< User-provided entry point for the stream. 
  AVFormatContext     *input_context;
 
public:
  Frame       setupframe;
  Frame       out_frame;
  long int    duration;
  long int    reftime; 
  long int    frame_mstimestamp_;    ///< Millisecond timestamp of the current available frame.  Note the convention: _ means that this is in frame time
  long int    stream_mstimestamp_;   ///< Millisecond timestamp: where the stream is currently.  -1 means nowhere (must seek first).
  FileState   state;
  AVPacket    *avpkt;
  std::vector<FrameType> frame_types;
  
public: // getters
  SlotNumber getSlot() {return slot;}
  
public:
  void setRefMstime(long int ms_streamtime_);
  void seek(long int ms_streamtime_);
  void play();
  void stop();
  void pullNextFrame(long int target_mstimestamp, long int &timeout, bool &reached);
  void pullFrames(long int target_mstimestamp);
  long int getNextTimestamp();
  bool playOrSeek();
  void stopSeek();
};


/** Used by SignalContext to carry info in the signal */
struct FileContext { // <pyapi>
  std::string    filename;        ///< incoming: the filename                                      // <pyapi>
  SlotNumber     slot;            ///< incoming: a unique stream slot that identifies this stream  // <pyapi>
  FrameFilter*   framefilter;     ///< incoming: the frames are feeded into this FrameFilter       // <pyapi>
  long int       seektime_;       ///< incoming: used by signal seek_stream                        // <pyapi>
  long int*      duration;        ///< outgoing: duration of the stream                            // <pyapi>
  long int*      mstimestamp;     ///< outgoing: current position of the stream (stream time)      // <pyapi>
  FileState      status;          ///< outgoing: status of the file                                // <pyapi>
}; // <pyapi>

// {"kokkelis.mkv", 1, framefilter, duration, mstimestamp}  


class FileThread : public Thread { // <pyapi>

  /** Characteristic signals for the FileThread.
   * 
   * These signals map directly into methods with the same names
   * 
   */
  enum class Signals {
    none,
    exit,                 
    open_stream,
    close_stream,
    seek_stream,
    play_stream,
    stop_stream,
    get_state            // query information about the stream
  };
  
  /** Identifies the information the signals FileThread::Signals carry.  Encapsulates a FileContext instance.
   *
   */
  struct SignalContext {
    Signals     signal;
    FileContext *file_context; // pointer cause we have return values
  };

  
public:                                                // <pyapi>
  /** Default constructor
   * 
   * @param name          Thread name
   * @param n_max_slots   Maximum number of connections (each Connection instance is placed in a slot)
   * 
   */
  FileThread(const char* name, int core_id=-1);        // <pyapi>
  ~FileThread();                                       // <pyapi>
  
protected: // redefinitions
  std::deque<SignalContext> signal_fifo;    ///< Redefinition of signal fifo (Thread::signal_fifo is now hidden from usage) 
  
protected:
  std::vector<FileStream*> slots_;        ///< A constant sized vector.  Keep the books for each stream.
  //std::map<SlotNumber,FileStream*> active_slots;
  std::list<SlotNumber>   active_slots;
  bool loop;                              ///< Controls the execution of the main loop
  std::list<FileStream*>  streamlist;     ///< FileStream s that have frames to be presented are queued here
  
public: // redefined virtual functions
  void run();
  void preRun();
  void postRun();
  /** @copydoc Thread::sendSignal */
  void sendSignal(SignalContext signal_ctx);         ///< Must be explicitly *redefined* just in case : Thread::SignalContext has been changed to LiveThread::SignalContext
  void sendSignalAndWait(SignalContext signal_ctx); 
  
protected:
  void handleSignals();
  
private: // internal
  int  safeGetSlot          (SlotNumber slot, FileStream*& stream);
  void openFileStream       (FileContext &file_ctx);
  void closeFileStream      (FileContext &file_ctx);
  void seekFileStream       (FileContext &file_ctx);
  void playFileStream       (FileContext &file_ctx);
  void stopFileStream       (FileContext &file_ctx);
  
public: // *** C & Python API *** .. these routines go through the convar/mutex locking                                // <pyapi>
  void closeFileStreamCall      (FileContext &file_ctx); ///< API method: registers a stream                                // <pyapi> 
  void openFileStreamCall       (FileContext &file_ctx); ///< API method: de-registers a stream                             // <pyapi>
  void seekFileStreamCall       (FileContext &file_ctx); ///< API method: seek to a certain point                           // <pyapi>
  void playFileStreamCall       (FileContext &file_ctx); ///< API method: starts playing the stream and feeding frames      // <pyapi>
  void stopFileStreamCall       (FileContext &file_ctx); ///< API method: stops playing the stream and feeding frames       // <pyapi>
  void stopCall();                                  ///< API method: stops the LiveThread                              // <pyapi>
}; // <pyapi>


