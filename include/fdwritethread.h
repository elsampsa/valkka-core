#ifndef fdwriterthread_HEADER_GUARD
#define fdwriterthread_HEADER_GUARD
/*
 * fdwriterthread.h : A general thread that write frames into something described by a file descriptor
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
 *  @file    fdwriterthread.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.3.5 
 *  
 *  @brief
 */ 

#endif

#include <sys/select.h>
#include "thread.h"
#include "framefilter.h"
#include "framefifo.h"


// (1) A data structure for describing the outgoing connection

/** Describes an outgoing file descriptor connection
 * 
 */
struct FDWriteContext {                                                                           // <pyapi>
    FDWriteContext() {}                                                                           // <pyapi>
    FDWriteContext(int fd, SlotNumber slot) : fd(fd), slot(slot) {}                               // <pyapi>
    int                 fd;              ///< file descriptor                                     // <pyapi>
    SlotNumber          slot;            ///< A unique stream slot that identifies this stream    // <pyapi>
};                                                                                                // <pyapi>

inline std::ostream& operator<< (std::ostream& os, const FDWriteContext& ctx) {
  // https://stackoverflow.com/questions/4571611/making-operator-virtual
  os << "<FDWriteContext : slot = " << ctx.slot << " / fd = " << ctx.fd << " >";
  return os;
}



// (2) A class for handling the outgoing connection.  Used internally by the FDWriteThread.

class FDWrite {
  
public:
    FDWrite(FrameFifo& fifo, const FDWriteContext& ctx);  ///< Default constructor
    virtual ~FDWrite(); ///< Default virtual destructor

public: // init'd at constructor time
    const FDWriteContext     &ctx;     ///< Identifies the connection type, stream address, etc.
    FrameFifo                &fifo;    ///< Outgoing Frames are finally recycled here
    std::deque<Frame*>       internal_fifo;
    
};



// (3) Define the communication with the Thread

/** Information sent with a signal to FDWriteThread
 * 
 */
struct FDWriteSignalPars {                                
    ///< Identifies the stream                             
    FDWriteContext fd_write_ctx;
}; 


/** All possible signals for FDWriteThread
 * 
 */
enum class FDWriteSignal {
    none,
    exit,
    register_stream,
    deregister_stream
};


/** Encapsulate data sent to FDWriteThread with a SignalFrame
 * 
 * 
 */
struct FDWriteSignalContext {
    FDWriteSignal         signal;
    FDWriteSignalPars     pars;
};




// (4) The Thread class

/** File Descriptor Writer Thread
 * 
 * - Receives frames into an incoming queue
 * - Maps the frames, using slot numbers, to file descriptors.  A file descriptor typically refers to a socket.
 * - This class also serves as a model for a generic thread that writes the frames somewhere (to a file, to a socket, etc.)
 * 
 */
class FDWriteThread : public Thread {                                                 // <pyapi>
  
public:                                                                               // <pyapi>
    /** Default constructor
    * 
    * @param name          Thread name
    * 
    */
    FDWriteThread(const char* name, FrameFifoContext fifo_ctx = FrameFifoContext());  // <pyapi>
    virtual ~FDWriteThread();                                                         // <pyapi>
        
protected: // frame input
    FDFrameFifo             infifo;           ///< Incoming frames (also signal frames) are read from here
    FifoFrameFilter         infilter;         ///< Write incoming frames here // TODO: add a chain of correcting FrameFilter(s)
    BlockingFifoFrameFilter infilter_block;   ///< Incoming frames can also be written here.  If stack runs out of frames, writing will block
    std::vector<FDWrite*>   slots_;           ///< For fast, pointer-arithmetic-based indexing of the slots
    std::list<FDWrite*>     fd_writes;        ///< For iterating over the FDWrite entries
    fd_set                  write_fds, read_fds; ///< File descriptor sets used by select
    int                     nfds;             ///< Max file descriptor number
    struct timeval          timeout;
    
    
public: // redefined virtual functions
    void run();
    void preRun();
    void postRun();
    void preJoin();
    void postJoin();
      
protected:
    void handleSignal(const FDWriteSignalContext &signal_ctx); ///< Handle an individual signal.  
    
private: // internal
    void setMaxFD         ();
    int  safeGetSlot      (const SlotNumber slot, FDWrite*& fd_write);  
    void registerStream   (const FDWriteContext &ctx);
    void deregisterStream (const FDWriteContext &ctx);
    
public: // *** C & Python API *** .. these routines go through the condvar/mutex locking                                      // <pyapi>
    // inbound streams
    void registerStreamCall   (const FDWriteContext &ctx); ///< API method: registers a stream                                // <pyapi> 
    void deregisterStreamCall (const FDWriteContext &ctx); ///< API method: de-registers a stream                             // <pyapi>
    void requestStopCall();
    FifoFrameFilter &getFrameFilter();                     ///< API method: get filter for sending frames                     // <pyapi>
};                                                                                                                            // <pyapi>
    




