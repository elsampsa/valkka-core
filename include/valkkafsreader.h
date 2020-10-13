#ifndef valkkafsreader_HEADER_GUARD
#define valkkafsreader_HEADER_GUARD
/*
 * valkkafsreader.h :
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
 *  @file    valkkafsreader.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.0.1 
 *  
 *  @brief
 */ 

#include "common.h"
#include "thread.h"
#include "framefifo.h"
#include "framefilter.h"
#include "valkkafs.h"
#include "rawrite.h"

/** ValkkaFS reader thread
 * 
 * - Reads frames from a ValkkaFS device file and writes them (one block at a time) to output framefilter
 * - Frames are requested on per-block basis
 * 
 *  This thread just knows how to send a block of frames, so its sending "stateless stream"
 * 
 *  That means that there's no notion of stream start = no SetupFrames are sent
 * 
 *  SetupFrames are typically added to the stream later on the filterchain, for example, using FileCacherThread.
 * 
 */
class ValkkaFSReaderThread : public Thread {                                                // <pyapi>

public:                                                                                     // <pyapi>
    ValkkaFSReaderThread(const char *name, ValkkaFS &valkkafs, FrameFilter &outfilter, FrameFifoContext fifo_ctx=FrameFifoContext(10), bool o_direct = false);     // <pyapi>
    ~ValkkaFSReaderThread();                                                                // <pyapi>

protected:
    ValkkaFS                        &valkkafs;
    FrameFilter                     &outfilter;
    std::map<IdNumber, SlotNumber>  id_to_slot;
    // std::fstream                    filestream;
    RawReader                       raw_reader;
    
protected: // frame input
    FrameFifo               infifo;           ///< Incoming frames are read from here.  The stream is "stateless", so you have to add SetupFrames (for example, by using FileCacherThread)
    FifoFrameFilter         infilter;         ///< Write incoming frames here.  Only for SignalFrames.

protected: // Thread member redefinitions
    std::deque<ValkkaFSReaderSignalContext> signal_fifo;   ///< Redefinition of signal fifo.
  
public: // redefined virtual functions
    void run();
    void preRun();
    void postRun();
    void sendSignal(ValkkaFSReaderSignalContext signal_ctx);    ///< Insert a signal into the signal_fifo
      
protected:
    void handleSignal(ValkkaFSReaderSignalContext &signal_ctx); ///< Handle an individual signal.  Signal can originate from the frame fifo or from the signal_fifo deque
    void handleSignals();                                       ///< Call ValkkaFSReaderThread::handleSignal for every signal in the signal_fifo

protected:
    void setSlotId(SlotNumber slot, IdNumber id);
    void unSetSlotId(IdNumber id);
    void clearSlotId();
    void reportSlotId();
    void pullBlocks(std::list<std::size_t> block_list);
    
// API
public:                                                       // <pyapi>
    FifoFrameFilter &getFrameFilter();                        // <pyapi>
    /** Set a id number => slot mapping */
    void setSlotIdCall(SlotNumber slot, IdNumber id);         // <pyapi>
    /** Clear a id number => slot mapping */
    void unSetSlotIdCall(IdNumber id);                    // <pyapi>
    /** Clear all id number => slot mappings */
    void clearSlotIdCall();                                   // <pyapi>
    /** Print id number => slot mappings */
    void reportSlotIdCall();                                  // <pyapi>
    void pullBlocksCall(std::list<std::size_t> block_list);
    /** Request blocks with python list */
    void pullBlocksPyCall(PyObject *pylist);                  // <pyapi>
    void requestStopCall();                                   // <pyapi>
};                                                            // <pyapi>




#endif
