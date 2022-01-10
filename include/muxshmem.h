#ifndef muxshmem_HEADER_GUARD
#define muxshmem_HEADER_GUARD
/*
 * muxshmem.h :
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
 *  @file    muxshmem.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.3.0 
 *  
 *  @brief
 */ 

#include "sharedmem.h"


class FragMP4SharedMemSegment : public SharedMemSegment {

public:
    FragMP4SharedMemSegment(const char* name, std::size_t n_bytes, bool is_server=false);
    virtual ~FragMP4SharedMemSegment();
    
protected:
    virtual void serverInit();       ///< Uses shmem_open with write rights.  used by the constructor if is_server=true.  Init with correct metadata serialization.
    virtual bool clientInit();       ///< Uses shmem_open with read-only rights.  Init with correct metadata serialization.
    virtual void serverClose();      ///< Erases the shmem segment.  used by the constructor if is_server=true
    virtual void clientClose();
    
public:
    virtual std::size_t getSize();                                    ///< Client: return metadata = the size of the payload (not the maximum size).  Uses SharedMemSegment::getMeta.
    virtual void put(std::vector<uint8_t> &inp_payload, void* meta_); ///< typecast void to std::size_t
    virtual void put(uint8_t* buf_, void* meta_);
    virtual void copyMetaFrom(void *meta_);       ///< Dereference metadata pointer correctly and copy the contents into this memory segment's metadata
    virtual void copyMetaTo(void *meta_);         ///< Dereference metadata pointer correctly and copy the contents from this memory segment's metadata
  
public:
    void putMuxFrame(MuxFrame *f);

public:
    FragMP4Meta *meta;
};


class FragMP4SharedMemRingBuffer : public SharedMemRingBufferBase { // <pyapi>

public:                                                     // <pyapi>
    FragMP4SharedMemRingBuffer(const char* name, int n_cells, std::size_t n_size, // <pyapi>
        int mstimeout=0, bool is_server=false); // <pyapi>
    virtual ~FragMP4SharedMemRingBuffer(); // <pyapi>

public:
    void serverPushMuxFrame(MuxFrame *f);
    // ..takes meta_blob 
  
public:                              // <pyapi>
// TODO
    PyObject* clientPullPy();                                     // <pyapi>
    // bool serverPushPyMux(PyObject *po, SlotNumber slot, long int mstimestamp); // <pyapi>
};                                                                // <pyapi>



class FragMP4ShmemFrameFilter : public FrameFilter { // <pyapi> 
  
public: // <pyapi>
    FragMP4ShmemFrameFilter(const char* name, int n_cells, std::size_t n_size, int mstimeout=0); // <pyapi>
    // ~FragMP4ShmemFrameFilter(); // <pyapi>
  
protected: // initialized at constructor                                                  
    FragMP4SharedMemRingBuffer shmembuf;
  
protected:
    virtual void go(Frame* frame);

public:                                                                                   // <pyapi>
    void useFd(EventFd &event_fd);                                                        // <pyapi>
}; // <pyapi>


#endif
