#ifndef valkkafs_HEADER_GUARD
#define valkkafs_HEADER_GUARD
/*
 * valkkafs.h :
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
 *  @file    valkkafs.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.8.0 
 *  
 *  @brief
 */ 

#include "thread.h"
#include "framefilter.h"
#include "framefifo.h"

#include "Python.h"
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

/*
#include "boost/python/numpy.hpp"
namespace p = boost::python;
namespace np = boost::python::numpy;
// https://www.boost.org/doc/libs/1_64_0/libs/python/doc/html/numpy/reference/ndarray.html
// https://github.com/ndarray/Boost.NumPy/blob/master/libs/numpy/example/simple.cpp
// let's not use boost..
*/

/** Book-keeping for ValkkaFS
 * 
 * @param device_file   File where payload is written.  Can be /dev/sdb etc., or just a plain file with some space reserved
 * @param block_file    Book-keeping of the blocks in the device file
 * @param blocksize     Size of a single block in bytes
 * @param n_blocks      Size of the device (or the part we want to use) in bytes
 * 
 * - At python, maintain a json file with these parameters and with the current_row (i.e. block) 
 * - .. current_row is informed by using callbacks to the python side
 * - At python, use also read() and setCurrentBlock(n_block)
 * - At python, use also clear() and clearDevice() if needed
 * 
 * Actual reading and writing of frames are done by other classes
 * 
 * 
 * device_id (long uint) subsession_index (int) mstimestamp (long int) media_type (AVMediaType) codec_id (AVCodecId) size (std::size_t) payload (char)
 * 
 * 
 * 
 */
class ValkkaFS {        // <pyapi>

public:                 // <pyapi>
    ValkkaFS(const char *device_file, const char *block_file, std::size_t blocksize, std::size_t n_blocks); // <pyapi>
    ~ValkkaFS();        // <pyapi>

protected:
    std::string     device_file;
    std::string     block_file;
    std::size_t     blocksize;
    std::size_t     n_blocks;
    std::size_t     device_size;
    std::vector<long int> tab;      ///< Blocktable
    // std::vector<long int> tab2;     ///< Copy of the blocktable in a numpy array
    // PyArray_Descr   *descr;
    // PyArrayObject   *arr;
    std::mutex      mutex;
    long int        col_0;          ///< Current column 0 value (max keyframe timestamp)
    long int        col_1;          ///< Current column 1 value (max anyframe timestamp)
    std::size_t     current_row;    ///< Row number (block) that's being written
    std::size_t     prev_row;       ///< Previous row number (block)
    PyObject        *pyfunc;        ///< Python callback that's triggered at block write
    
protected:
    const static std::size_t  n_cols = 2;
    
protected:
    std::size_t     ind(std::size_t i, std::size_t j);                  ///< first index: block number (row), second index: column
    
public:
    void            setVal(std::size_t i, std::size_t j, long int val); ///< set tab's value at block i, row j
    long int        getVal(std::size_t i, std::size_t j);
    std::size_t     getBlockSeek(std::size_t n_block);
    std::size_t     getCurrentBlockSeek();
    
public:                                  // <pyapi>
    std::size_t get_n_blocks();          // <pyapi>
    std::size_t get_n_cols();            // <pyapi>
    
    void            dump();              // <pyapi>
    void            read();              // <pyapi>
    std::string     getDevice();         // <pyapi>
    std::size_t     getDeviceSize();     // <pyapi>
    void            clearDevice();       // <pyapi>
    void            clearTable();        // <pyapi>
    
    /** Used by a writer class to inform that a new block has been written */
    void writeBlock();                           // <pyapi>
    
    /** Used by a writer class to inform that a non-key frame has been written */
    void markFrame(long int mstimestamp);        // <pyapi>

    /** Used by a writer class to inform that a key frame has been written */
    void markKeyFrame(long int mstimestamp);     // <pyapi>
    
    /** Set block number that's being written */
    void setCurrentBlock(std::size_t n_block);      // <pyapi>
    
    /** Set a python callable that's being triggered when a new block is written */
    void setBlockCallback(PyObject* pobj);          // <pyapi>
    
    /** Copy blocktable to a given numpy array
     * 
     * @param   numpy array
     * 
     * The array must be created on the python side with:
     * 
     * a = numpy.zeros((v.get_n_blocks(), v.get_n_cols()),dtype=numpy.int_)
     * 
     */
    void setArrayCall(PyObject *pyobj);  // <pyapi>
};                                       // <pyapi>


/** Writes frames to a file
 * 
 * - Reads frames from infifo
 * - Writes them to the disk as they arrive
 * - Uses a ValkkaFS instance for book-keeping
 * 
 */
class ValkkaFSWriterThread : public Thread {         // <pyapi>

public:                                              // <pyapi>
    ValkkaFSWriterThread(const char *name, ValkkaFS &valkkafs, FrameFifoContext fifo_ctx=FrameFifoContext());  // <pyapi>
    ~ValkkaFSWriterThread();                                                               // <pyapi>

protected:
    ValkkaFS                            &valkkafs;
    std::ofstream                       filestream;
    std::map<SlotNumber, IdNumber>      slot_to_id;  ///< Map from slot numbers to ids
    
protected: // frame input
    FrameFifo               infifo;           ///< Incoming frames are read from here
    FifoFrameFilter         infilter;         ///< Write incoming frames here // TODO: add a chain of correcting FrameFilter(s)
    BlockingFifoFrameFilter infilter_block;   ///< Incoming frames can also be written here.  If stack runs out of frames, writing will block

protected: // Thread member redefinitions
    std::deque<ValkkaFSWriterSignalContext> signal_fifo;   ///< Redefinition of signal fifo.
  
public: // redefined virtual functions
    void run();
    void preRun();
    void postRun();
    void sendSignal(ValkkaFSWriterSignalContext signal_ctx);    ///< Insert a signal into the signal_fifo
      
protected:
    void handleSignal(ValkkaFSWriterSignalContext &signal_ctx); ///< Handle an individual signal.  Signal can originate from the frame fifo or from the signal_fifo deque
    void handleSignals();                                       ///< Call ValkkaFSWriterThread::handleSignal for every signal in the signal_fifo

// API
public:                                                       // <pyapi>
    FifoFrameFilter &getFrameFilter();                        // <pyapi>
    FifoFrameFilter &getBlockingFrameFilter();                // <pyapi>
    /** Set a slot => id number mapping
    */
    void setSlotIdCall(SlotNumber slot, IdNumber id);         // <pyapi>
    /** Seek to a certain block
    */
    void seekCall(std::size_t n_block);                       // <pyapi>
    void requestStopCall();                                   // <pyapi>
};                                                            // <pyapi>


class ValkkaFSReaderThread : public Thread {
    
public:
    ValkkaFSReaderThread(const char *name, ValkkaFS &valkkafs);
    ~ValkkaFSReaderThread();

private:
    ValkkaFS    &valkkafs;
    
    
    /*
    ValkkaFSReadThread(ValkkaFS valkkafs) / or "ValkkaFSDumpThread"

    pullBlocks(std::list)     Dump all blocks in the list to output filters
    
    **Python API**
    
    getValkkaFS()
    pullBlocksCall(PyObject)
    RegisterOutputCall(context with id and cam id)
    deRegisterOutputCall(cam id)
    
    ? put here playCall, seekCall, etc.
    =>
    */
};
    


#endif
