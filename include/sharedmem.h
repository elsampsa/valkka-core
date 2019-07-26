#ifndef sharedmem_HEADER_GUARD
#define sharedmem_HEADER_GUARD
/*
 * sharedmem.h : Posix shared memory segment server/client management, shared memory ring buffer synchronized using posix semaphores.
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
 *  @file    sharedmem.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.13.1 
 *  
 *  @brief Posix shared memory segment server/client management, shared memory ring buffer synchronized using posix semaphores.
 */ 


#include "common.h"
#include <sys/shm.h>
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <semaphore.h> // semaphores
#include "framefilter.h"
#include "Python.h"

/** Shared memory segment with metadata (the segment size)
 * 
 * First, instantiate a shared memory segment in the server side, with is_server=true
 *
 * Second, from another process, instantiate shared memory segment with the same name, but with is_server=false
 *
 * Never instantiate both server and client side from the same process.
 * 
 * This is a virtual class.  Child classes should define metadata and its serialization.  This is done in serverInit, serverClose, clientInit and clientClose
 * 
 * @ingroup shmem_tag
 */

// https://stackoverflow.com/questions/115703/storing-c-template-function-definitions-in-a-cpp-file
class SharedMemSegment {

public:
  /** Default constructor
   * 
   * @param name       String identifying the shmem segment in the posix shmem system
   * @param n_bytes    Size of the byte buffer
   * @param is_server  In the server side, instantiate with true, in client side, instantiate with false
   * 
   */
  SharedMemSegment(const char* name, std::size_t n_bytes, bool is_server=false);
  /** Default destructor */
  virtual ~SharedMemSegment();
  
protected:
    virtual void serverInit() = 0;      ///< Server: Uses shmem_open with write rights.  used by the constructor if is_server=true.  Init with correct metadata serialization.
    virtual bool clientInit() = 0;      ///< Client: Uses shmem_open with read-only rights.  Init with correct metadata serialization.
    virtual void serverClose() = 0;     ///< Erases the shmem segment.  used by the constructor if is_server=true
    virtual void clientClose() = 0;
  
public:
    virtual std::size_t getSize() = 0;        ///< Client: return size of payload
    virtual void put(std::vector<uint8_t> &inp_payload, void *meta_) = 0; ///< Server: copy byte chunk into payload accompanied with metadata.  Corrent typecast in child class methods.
    virtual void copyMetaFrom(void *meta_) = 0; ///< Dereference metadata pointer correctly and copy the contents into this memory segment's metadata
    virtual void copyMetaTo(void *meta_)   = 0; ///< Dereference metadata pointer correctly and copy the contents from this memory segment's metadata
  
protected:
    std::string   name;         ///< Name to identify the posix objects
    std::string   payload_name; ///< Name to identify the posix memory mapped file
    std::string   meta_name;    ///< Name to identify the posix memory mapped file for the metadata
    bool          is_server;    ///< Client or server process?
    void          *ptr,*ptr_;   ///< Raw pointers to shared memory
    bool          client_state; ///< Was the shmem acquisition succesfull?
    
public:
    uint8_t     *payload;   ///< Pointer to payload
    // void        *meta;      ///< Metadata (define in child classes);
    std::size_t n_bytes;    ///< Maximum size of the payload (this much is reserved)
    
public: // client
    void init();                           ///< Must be called after construction.  Reserves shmem payload and correct metadata (depending on the subclass)
    void close_();                         ///< Must be called before destruction. Releases correct amount of metadata bytes (depending on the subclass)
    bool getClientState();                 ///< Was the shmem acquisition succesfull?
};


/** Shared mem segment with simple metadata : just the payload length
 * 
 * 
 */
class SimpleSharedMemSegment : public SharedMemSegment {
   
public:
    SimpleSharedMemSegment(const char* name, std::size_t n_bytes, bool is_server=false);
    virtual ~SimpleSharedMemSegment();

protected:
    virtual void serverInit();       ///< Uses shmem_open with write rights.  used by the constructor if is_server=true.  Init with correct metadata serialization.
    virtual bool clientInit();       ///< Uses shmem_open with read-only rights.  Init with correct metadata serialization.
    virtual void serverClose();      ///< Erases the shmem segment.  used by the constructor if is_server=true
    virtual void clientClose();
    
public:
    virtual std::size_t  getSize();               ///< Client: return metadata = the size of the payload (not the maximum size).  Uses SharedMemSegment::getMeta.
    virtual void put(std::vector<uint8_t> &inp_payload, void* meta_); ///< typecast void to std::size_t
    virtual void copyMetaFrom(void *meta_);       ///< Dereference metadata pointer correctly and copy the contents into this memory segment's metadata
    virtual void copyMetaTo(void *meta_);         ///< Dereference metadata pointer correctly and copy the contents from this memory segment's metadata
  
public:
    std::size_t  *meta;
    
public:
    void         put(std::vector<uint8_t> &inp_payload);    ///< for legacy API
    void         putAVRGBFrame(AVRGBFrame *f);              ///< Copy from AVFrame->data directly.  Only metadata used: payload size
};


/** A seriazable metadata object
 * 
 * 
 */
struct RGB24Meta {                                  // <pyapi>
    std::size_t size; ///< Actual size copied       // <pyapi>
    int width;                                      // <pyapi>
    int height;                                     // <pyapi>
    SlotNumber slot;                                // <pyapi>
    long int mstimestamp;                           // <pyapi>
};                                                  // <pyapi>


/** A Shmem segment describing an RGB24 frame
 * 
 * - Metadata includes width, height, slot, mstimestamp
 * 
 */
class RGB24SharedMemSegment : public SharedMemSegment {
    
public:
    RGB24SharedMemSegment(const char* name, int width, int height, bool is_server=false);
    virtual ~RGB24SharedMemSegment();
    
protected:
    virtual void serverInit();       ///< Uses shmem_open with write rights.  used by the constructor if is_server=true.  Init with correct metadata serialization.
    virtual bool clientInit();       ///< Uses shmem_open with read-only rights.  Init with correct metadata serialization.
    virtual void serverClose();      ///< Erases the shmem segment.  used by the constructor if is_server=true
    virtual void clientClose();
    
public:
    virtual std::size_t getSize();                                    ///< Client: return metadata = the size of the payload (not the maximum size).  Uses SharedMemSegment::getMeta.
    virtual void put(std::vector<uint8_t> &inp_payload, void* meta_); ///< typecast void to std::size_t
    virtual void copyMetaFrom(void *meta_);       ///< Dereference metadata pointer correctly and copy the contents into this memory segment's metadata
    virtual void copyMetaTo(void *meta_);         ///< Dereference metadata pointer correctly and copy the contents from this memory segment's metadata
  
public:
    RGB24Meta    *meta;
    
public:
    void         putAVRGBFrame(AVRGBFrame *f);                        ///< Copy from AVFrame->data directly.  Copies more metadata: width, height, slot & mstimestamp
};


/** Interprocess shared memory ring buffer synchronized with posix semaphores.
 * 
 * Create first a server instance.  Then, in another process, create the client instance with the same name, but with is_server=false.
 *
 * This class is only for multiprocessing.  Don't never ever use it in the context of multithreading.  And don't start the server and client side from the same process.
 * 
 * Don't expect this shmem ring buffer to work for high-throughput media streaming.  It's good for sending a few frames per second between multiprocesses.  For high-throughput cases, use multithreading instead.
 * 
 * @ingroup shmem_tag
 */
class SharedMemRingBufferBase { // <pyapi>

public: // <pyapi>
    /** Default constructor
    * 
    * @param name        Name of the ring buffer.  This name is used as unique identifier for the posix semaphores and shmem segments.  Don't use weird characters.
    * @param n_cells     Number of cells in the ring buffer
    * @param n_bytes     Size of each ring buffer cell in bytes
    * @param mstimeout   Semaphore timeout in milliseconds.  SharedMemRingBuffer::clientPull returns after this many milliseconds even if data was not received.  Default 0 = waits forever.
    * @param is_server   Set this to true if you are starting this from the server multiprocess
    * 
    * Child classes should reserve correct SharedMemSegment type
    * 
    */
    SharedMemRingBufferBase(const char* name, int n_cells, std::size_t n_bytes, int mstimeout=0, bool is_server=false); // <pyapi>
    /** Default destructor */
    virtual ~SharedMemRingBufferBase(); // <pyapi>

protected: // at constructor init list
    std::string  name;
    int          n_cells, n_bytes; ///< Parameters defining the shmem ring buffer (number, size)
    int          mstimeout;        ///< Semaphore timeout in milliseconds
    bool         is_server;        ///< Are we on the server side or not?
    
protected: // posix semaphores and mmap'd files
    sem_t         *sema, *flagsema;///< Posix semaphore objects (semaphore counter, semaphore for the overflow flag)
    std::string   sema_name;       ///< Name to identify the posix semaphore counter
    std::string   flagsema_name;   ///< Name to identify the posix semaphore used for the overflow flag
    int  index;                    ///< The index of cell that has just been written.  Remember: server and client see their own copies of this
    struct timespec ts;            ///< Timespec for semaphore timeouts
    
public:
    std::vector<SharedMemSegment*> shmems; ///< Shared memory segments

protected: // internal methods - not for the api user
    void  setFlag();       ///< Server: call this to indicate a ring-buffer overflow 
    bool  flagIsSet();     ///< Client: call this to see if there has been a ring-buffer overflow
    void  clearFlag();     ///< Client: call this after handling the ring-buffer overflow
    int   getFlagValue();  ///< Used by SharedMemoryRingBuffer::flagIsSet()
    void  zero();          ///< Force reset.  Semaphore value is set to 0
    
public:
  int   getValue();       ///< Returns the current index (next to be read) of the shmem buffer
  bool  getClientState(); ///< Are the shmem segments available for client?
  
public: // server side routines - call these only from the server           
    /** Copies payload to ring buffer
    * 
    * @param inp_payload : std::vector bytebuffer (passed by reference)
    * 
    * After copying the payload, releases (increases) the semaphore.
    */
    void serverPush(std::vector<uint8_t> &inp_payload, void* meta);
    
    
public: // client side routines - call only from the client side // <pyapi>
    /** Returns the index of SharedMemoryRingBuffer::shmems that was just written.
    * 
    * @param index_out : returns the index of the shmem ringbuffer just written
    * @param size_out  : returns the size of the payload written to the shmem ringbuffer
    * 
    * returns true if data was obtained, false if semaphore timed out
    * 
    */
    /** shmem index, metadata object to be filled */
    bool clientPull(int &index_out, void* meta);                // <pyapi>
    /** multithreading version: releases GIL */
    bool clientPullThread(int &index_out, void* meta);          // <pyapi>
    PyObject *getBufferListPy();                                // <pyapi>
}; // <pyapi>



class SharedMemRingBuffer : public SharedMemRingBufferBase { // <pyapi>

public: // <pyapi>
    SharedMemRingBuffer(const char* name, int n_cells, std::size_t n_bytes, int mstimeout=0, bool is_server=false); // <pyapi>
    /** Default destructor */
    virtual ~SharedMemRingBuffer(); // <pyapi>
    
public:                                                     // <pyapi>
    void serverPush(std::vector<uint8_t> &inp_payload);     // <pyapi>
    bool clientPull(int &index_out, int &size_out);         // <pyapi>
};                                                          // <pyapi>



/** SharedMemRingBuffer for AVRGBFrame.
 * 
 * @ingroup shmem_tag
 */
class SharedMemRingBufferRGB : public SharedMemRingBufferBase { // <pyapi>

public:                                                     // <pyapi>
    /** Default ctor */
    SharedMemRingBufferRGB(const char* name, int n_cells, int width, int height, int mstimeout=0, bool is_server=false); // <pyapi>
    /** Default destructor */
    ~SharedMemRingBufferRGB(); // <pyapi>

protected:
    int width, height;

public:
    void serverPushAVRGBFrame(AVRGBFrame *f);
  
public:                              // <pyapi>
    /** Returns a python tuple of metadata
     * (index, width, height, slot, timestamp)
     */
    // PyObject* clientPullPy();                                  // <pyapi>
    /** Legacy support */
    bool clientPull(int &index_out, int &size_out);               // <pyapi>
    /** Pulls payload and extended metadata */
    bool clientPullFrame(int &index_out, RGB24Meta &meta);        // <pyapi>
    /** For multithreading (instead of multiprocessing) applications: releases python GIL */
    bool clientPullFrameThread(int &index_out, RGB24Meta &meta);  // <pyapi>
};                                                                // <pyapi>



/** This FrameFilter writes frames into a SharedMemRingBuffer
 * 
 * @ingroup filters_tag
 * @ingroup shmem_tag
 */
class ShmemFrameFilter : public FrameFilter {                                               // <pyapi>
  
public:                                                                                     // <pyapi>
  /** Default constructor
   * 
   * Creates and initializes a SharedMemoryRingBuffer.  Frames fed into this FrameFilter are written into that buffer.
   * 
   * @param   name. Important!  Identifies the SharedMemoryRingBuffer where the frames are being written
   * @param   n_cells. Number of shared memory segments in the ring buffer
   * @param   n_bytes. Size of shared memory segments in the ringbuffer
   * @param   mstimeout. Semaphore timeout wait for the ringbuffer
   * 
   */
  ShmemFrameFilter(const char* name, int n_cells, std::size_t n_bytes, int mstimeout=0);  // <pyapi>
  //~ShmemFrameFilter(); // <pyapi>

protected: // initialized at constructor                                                  
  //int                    n_cells;
  //std::size_t            n_bytes;
  //int                    mstimeout;
  SharedMemRingBuffer shmembuf;
  
protected:
  virtual void go(Frame* frame);
  
};                                                                                       // <pyapi>
  

/** Like ShmemFrameFilter.  Writes frames into SharedMemRingBufferRGB
 * 
 * @ingroup filters_tag
 * @ingroup shmem_tag
 */
class RGBShmemFrameFilter : public FrameFilter { // <pyapi> 
  
public: // <pyapi>
  /** Default constructor
   */
  RGBShmemFrameFilter(const char* name, int n_cells, int width, int height, int mstimeout=0); // <pyapi>
  //~RGBShmemFrameFilter(); // <pyapi>
  
protected: // initialized at constructor                                                  
  //int                    n_cells;
  //int                    width;
  //int                    height;
  //int                    mstimeout;
  SharedMemRingBufferRGB shmembuf;
  
protected:
  virtual void go(Frame* frame);
  
}; // <pyapi>
  
  
#endif
