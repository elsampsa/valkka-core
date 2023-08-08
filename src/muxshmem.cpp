/*
 * muxshmem.cpp :
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
 *  @file    muxshmem.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.5.2 
 *  
 *  @brief 
 */ 

#include "muxshmem.h"


FragMP4SharedMemSegment::FragMP4SharedMemSegment(const char* name, std::size_t n_bytes, bool is_server) : SharedMemSegment(name, n_bytes, is_server) {
    }

FragMP4SharedMemSegment::~FragMP4SharedMemSegment() {
    }
    
server_init(FragMP4SharedMemSegment, FragMP4Meta);
client_init(FragMP4SharedMemSegment, FragMP4Meta);
server_close(FragMP4SharedMemSegment, FragMP4Meta);
client_close(FragMP4SharedMemSegment, FragMP4Meta);
copy_meta_from(FragMP4SharedMemSegment, FragMP4Meta);
copy_meta_to(FragMP4SharedMemSegment, FragMP4Meta);

std::size_t FragMP4SharedMemSegment::getSize() {
    return meta->size;
}

void FragMP4SharedMemSegment::put(std::vector<uint8_t> &inp_payload, void* meta_) {
    *meta = *((FragMP4Meta*)(meta_));
    meta->size = std::min(inp_payload.size(), n_bytes);
    memcpy(payload, inp_payload.data(), meta->size);
}

void FragMP4SharedMemSegment::put(uint8_t* buf, void* meta_) {
    *meta = *((FragMP4Meta*)(meta_));
    meta->size = std::min(std::size_t(meta->size), n_bytes);
    memcpy(payload, buf, meta->size);
}


void FragMP4SharedMemSegment::putMuxFrame(MuxFrame *f) { // copy from AVFrame->data directly    
    if (f->meta_type != MuxMetaType::fragmp4) {
        std::cout << "FragMP4SharedMemSegment::putMuxFrame: needs MuxMetaType::fragmp4"
            << std::endl;
        return; 
    }
    FragMP4Meta* meta_ = (FragMP4Meta*)(f->meta_blob.data());
    *meta = *meta_;
    meta->size = std::min(f->payload.size(), n_bytes); // correct meta->size if there was more than allowed n_bytes
    memcpy(payload, f->payload.data(), meta->size);
}


FragMP4SharedMemRingBuffer::FragMP4SharedMemRingBuffer(const char* name, int n_cells, std::size_t n_size, 
    int mstimeout, bool is_server) : 
    SharedMemRingBufferBase(name, n_cells, n_size, mstimeout, is_server) {
    int i;
    for(i=0; i<n_cells; i++) {
        shmems.push_back(
        new FragMP4SharedMemSegment((name+std::to_string(i)).c_str(), n_size, is_server)
            );
    }
    for(auto it = shmems.begin(); it != shmems.end(); ++it) {
        (*it)->init(); // init reserves the shmem at the correct subclass
    }
}


FragMP4SharedMemRingBuffer::~FragMP4SharedMemRingBuffer() {
    // std::cout << "SharedMemRingBufferRGB: dtor" << std::endl;
    for(auto it = shmems.begin(); it != shmems.end(); ++it) {
        (*it)->close_();
        delete *it;
    }
}


void FragMP4SharedMemRingBuffer::serverPushMuxFrame(MuxFrame *f) {
  int i;
    
  if (getValue()>=n_cells) { // so, semaphore will overflow
    zero();
    std::cout << "FragMP4SharedMemRingBuffer: ServerPush: zeroed, value now="<<getValue()<<std::endl;
    index=-1;
    setFlag();
    std::cout << "FragMP4SharedMemRingBuffer: ServerPush: OVERFLOW "<<std::endl;
  }
  
  ++index;
  if (index >= n_cells) {
    index=0;
  }
  ( (FragMP4SharedMemSegment*)(shmems[index]) )->putMuxFrame(f);
  // std::cout << "RingBuffer: ServerPush: wrote to index "<<index<<std::endl;
  
  i=sem_post(sema);
  setEventFd();
}

/* // TODO: if we would like to share mp4 fragments from the python side through shmem
bool FragMP4SharedMemRingBuffer::serverPushPyRGB(PyObject *po, SlotNumber slot, long int mstimestamp) {
    Py_INCREF(po);
    int i;
    PyArrayObject *pa = (PyArrayObject*)po;

    if (PyArray_NDIM(pa) < 3) {
        std::cout << "RingBuffer: ServerPushPyRGB: incorrect dimensions" << std::endl;
        return false;
    }

    npy_intp *dims = PyArray_DIMS(pa);

    RGB24Meta meta_ = RGB24Meta();
    meta_.size = dims[0]*dims[1]*dims[2];
    meta_.width = dims[1]; 
    meta_.height = dims[0];  // slowest index (y) is the first
    meta_.slot = slot;
    meta_.mstimestamp = mstimestamp;
    
    uint8_t* buf = (uint8_t*)PyArray_BYTES(pa);

    if (getValue()>=n_cells) { // so, semaphore will overflow
        zero();
        std::cout << "RingBuffer: ServerPush: zeroed, value now="<<getValue()<<std::endl;
        index=-1;
        setFlag();
        std::cout << "RingBuffer: ServerPush: OVERFLOW "<<std::endl;
    }
    
    ++index;
    if (index >= n_cells) {
        index=0;
    }

    // std::cout << name << " push: meta_.size = " << meta_.size << std::endl;

    ( (RGB24SharedMemSegment*)(shmems[index]) )->put(buf, (void*)&meta_);
    // std::cout << "RingBuffer: ServerPush: wrote to index "<<index<<std::endl;
    
    Py_DECREF(po);

    i=sem_post(sema);
    setEventFd();
    return true;
}
*/


PyObject* FragMP4SharedMemRingBuffer::clientPullPy() {
    PyObject* tup = PyTuple_New(6);
    FragMP4Meta meta = FragMP4Meta();
    int index_out = 0;
    bool ok;
    Py_BEGIN_ALLOW_THREADS
    ok = SharedMemRingBufferBase::clientPull(index_out, (void*)(&meta));
    Py_END_ALLOW_THREADS
    if (!ok) {
        index_out = -1;
        PyTuple_SetItem(tup, 0, PyLong_FromLong((long)index_out));
    }
    else {
        PyTuple_SetItem(tup, 0, PyLong_FromLong((long)index_out));
        PyTuple_SetItem(tup, 1, PyLong_FromSsize_t(meta.size));
        PyTuple_SetItem(tup, 2, 
            PyLong_FromUnsignedLong((unsigned long)(meta.slot))); // unsigned short
        PyTuple_SetItem(tup, 3, PyLong_FromLong(meta.mstimestamp));
        // std::cout << "-->" << meta.name << "<--" << std::endl;
        PyTuple_SetItem(tup, 4, PyBytes_FromString(meta.name));
        PyTuple_SetItem(tup, 5, PyBool_FromLong(long(meta.is_first)));
        // put here whatever metadata we might
        // add in the future..
    }
    return tup;
}


FragMP4ShmemFrameFilter::FragMP4ShmemFrameFilter(const char* name, 
    int n_cells, std::size_t n_size, int mstimeout) : 
    FrameFilter(name,NULL), shmembuf(name, n_cells, n_size, mstimeout, true) {
}


void FragMP4ShmemFrameFilter::useFd(EventFd &event_fd) {
    shmembuf.serverUseFd(event_fd);
}


void FragMP4ShmemFrameFilter::go(Frame* frame) {
    if (frame->getFrameClass()!=FrameClass::mux) {
        std::cout << "FragMP4ShmemFrameFilter: go: ERROR: MuxFrame required" << std::endl;
        return;
    }
    MuxFrame *muxframe =static_cast<MuxFrame*>(frame);
    if (muxframe->meta_type != MuxMetaType::fragmp4) {
        std::cout << "FragMP4ShmemFrameFilter::go: needs MuxMetaType::fragmp4"
            << std::endl;
        return; 
    } 
    shmembuf.serverPushMuxFrame(muxframe);
}
