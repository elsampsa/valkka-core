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
 *  @version 0.18.0 
 *  
 *  @brief
 */ 

#include "common.h"
#include "thread.h"
#include "framefilter.h"
#include "framefifo.h"
#include "rawrite.h"
#include "logging.h"
#include "Python.h"

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
 * @param init          Clear the block_file even if it exists
 * 
 * - At python, maintain a json file with these parameters and with the current_row (i.e. block) 
 * - .. current_row is informed by using callbacks to the python side
 * - At python, use also setCurrentBlock(n_block)
 * - At python, use also clearTable() [and clearDevice()] if needed
 * - The last block that's currently being written (current_row) is overwritten from start when the writing starts the next time
 * 
 * Actual reading and writing of frames are done by other classes (ValkkaFSWriterThread and ValkkaFSReaderThread)
 * 
 * When reading frames, they are passed between processes typically like this:
 * 
 * ValkkaFSReaderThread => CacheStream => Decoder
 * 
 * - Frame serialization : device_id (long uint) subsession_index (int) mstimestamp (long int) media_type (AVMediaType) codec_id (AVCodecId) size (std::size_t) payload (char)
 * 
 * 
 * 
 */
class ValkkaFS {        // <pyapi>

public:                 // <pyapi>
    
    /** Default Constructor
     * 
     * @param   device_file   where the payload is written (a file or a disk device)
     * @param   block_file    where the filesystem (block) information is written.  A dump of blocktable
     * @param   blocksize     size of a block
     * @param   n_blocks      number of blocks
     * @param   init          true = init and dump blocktable to disk.  false = try to read blocktable from disk (default)
     */
    ValkkaFS(const char *device_file, const char *block_file, std::size_t blocksize, std::size_t n_blocks, bool init=false); // <pyapi>
    ~ValkkaFS();        // <pyapi>

protected:
    std::string     device_file;
    std::string     block_file;
    std::size_t     blocksize;
    std::size_t     n_blocks;
    bool            init;           ///< Clear the blocktable or not even if it exists
    std::size_t     device_size;
    std::vector<long int> tab;      ///< Blocktable
    std::fstream    os;             ///< Write handle to blocktable file
    
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
    
public: // getters
    const std::size_t   getBlockSize() {return this->blocksize;} // <pyapi>
    
public:                                  // <pyapi>
    std::size_t get_n_blocks();          // <pyapi>
    std::size_t get_n_cols();            // <pyapi>
    
    /** dump blocktable to disk.  Not thread safe. */
    void dumpTable_();
    /** dump single row of bloctable to disk.  Not thread safe. */
    void updateDumpTable_(std::size_t n_block);
    
    /** dump blocktable to disk */
    void            dumpTable();              // <pyapi>
    /** read blocktable from disk */
    void            readTable();          // <pyapi>
    /** returns device filename */
    std::string     getDevice();         // <pyapi>
    /** returns device file size */
    std::size_t     getDeviceSize();     // <pyapi>
    /** writes zero bytes to the device */
    void            clearDevice(bool writethrough=false, bool verbose=false);       // <pyapi>
    /** clears the blocktable and writes it to the disk */
    void            clearTable();        // <pyapi>
    /** returns maximum allowed frame size in bytes */
    std::size_t     maxFrameSize();      // <pyapi>
    /** print blocktable */
    void            reportTable(std::size_t from=0, std::size_t to=0, bool show_all=false);       // <pyapi>
    /** Used by a writer class to inform that a new block has been written
     * @param pycall   : Use the provided python callback function or not?  default = true
     * @param use_gil  : Acquire Python GIL or not? should be true, when evoked "autonomously" by this thread and false, when evoked from python.  default = true
     * 
     */
    void writeBlock(bool pycall=true, bool use_gil=true);   // <pyapi>
    
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


/** Analyzer tool for ValkkaFS
 * 
 * - Dump blocks to the terminal
 * 
 */
class ValkkaFSTool {                        // <pyapi>
    
public:                                     // <pyapi>
    ValkkaFSTool(ValkkaFS &valkkafs);       // <pyapi>
    ~ValkkaFSTool();                        // <pyapi>
    
protected:
    // std::fstream     is;
    RawReader        raw_reader;
    ValkkaFS         &valkkafs;
    
public:                                     // <pyapi>
    void dumpBlock(std::size_t n_block);    // <pyapi>
};                                          // <pyapi>



/** Writes frames to ValkkaFS
 * 
 * - Reads frames from infifo
 * - Writes them to the disk as they arrive
 * - Uses a ValkkaFS instance for book-keeping
 * 
 */
class ValkkaFSWriterThread : public Thread {         // <pyapi>

public:                                              // <pyapi>
    ValkkaFSWriterThread(const char *name, ValkkaFS &valkkafs, FrameFifoContext fifo_ctx=FrameFifoContext(), bool o_direct = false);  // <pyapi>
    ~ValkkaFSWriterThread();                                                               // <pyapi>

protected:
    ValkkaFS                            &valkkafs;
    // std::fstream                        filestream;
    RaWriter                            raw_writer;
    std::map<SlotNumber, IdNumber>      slot_to_id;  ///< Map from slot numbers to ids
    std::size_t                         bytecount;
    
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
    void preJoin();
    void postJoin();
    void sendSignal(ValkkaFSWriterSignalContext signal_ctx);    ///< Insert a signal into the signal_fifo
      
protected:
    void handleSignal(ValkkaFSWriterSignalContext &signal_ctx); ///< Handle an individual signal.  Signal can originate from the frame fifo or from the signal_fifo deque
    void handleSignals();                                       ///< Call ValkkaFSWriterThread::handleSignal for every signal in the signal_fifo

protected:
    void saveCurrentBlock(bool pycall=true, bool use_gil=true);
    void setSlotId(SlotNumber slot, IdNumber id);
    void unSetSlotId(SlotNumber slot);
    void clearSlotId();
    void reportSlotId();
    void seek(std::size_t n_block);
    
// API
public:                                                       // <pyapi>
    FifoFrameFilter &getFrameFilter();                        // <pyapi>
    FifoFrameFilter &getBlockingFrameFilter();                // <pyapi>
    /** Set a slot => id number mapping */
    void setSlotIdCall(SlotNumber slot, IdNumber id);         // <pyapi>
    /** Clear a slot => id number mapping */
    void unSetSlotIdCall(SlotNumber slot);                    // <pyapi>
    /** Clear all slot => id number mappings */
    void clearSlotIdCall();                                   // <pyapi>
    /** Print slot, id number mappings */
    void reportSlotIdCall();                                  // <pyapi>
    /** Seek to a certain block */
    void seekCall(std::size_t n_block);                       // <pyapi>
    void requestStopCall();                                   // <pyapi>
};                                                            // <pyapi>


#endif
