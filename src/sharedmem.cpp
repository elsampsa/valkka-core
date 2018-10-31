/*
 * sharedmem.cpp : Posix shared memory segment server/client management, shared memory ring buffer synchronized using posix semaphores.
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
 *  @file    sharedmem.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.8.0 
 *  
 *  @brief   Posix shared memory segment server/client management, shared memory ring buffer synchronized using posix semaphores.
 */ 

#include "tools.h"
#include "sharedmem.h"

SharedMemSegment::SharedMemSegment(const char* name, std::size_t n_bytes, bool is_server) : name(name), n_bytes(n_bytes), is_server(is_server), client_state(false) {
  payload_name=std::string("/")+name+std::string("_valkka_payload");
  meta_name   =std::string("/")+name+std::string("_valkka_meta");
  
  if (is_server) {
    serverInit();
  }
  else {
    client_state=clientInit();
  }
}
  

SharedMemSegment:: ~SharedMemSegment() {
  if (is_server) {
    serverClose();
  }
  else {
    clientClose();
  }
}


void SharedMemSegment::serverInit() {
  int fd, fd_, r, r_; 
  
  shm_unlink(payload_name.c_str()); // just in case..
  shm_unlink(meta_name.c_str());
  
  fd = shm_open(payload_name.c_str(), O_CREAT | O_EXCL | O_TRUNC | O_RDWR, 0600); // use r/w
  fd_= shm_open(meta_name.c_str(),    O_CREAT | O_EXCL | O_TRUNC | O_RDWR, 0600); // use r/w
  
  if (fd == -1 or fd_==-1) {
    perror("valkka_core: sharedmem.cpp: SharedMemSegment::serverInit: shm_open failed");
    exit(2);
  }
  
  // std::cout << "got shmem" << std::endl;
  
  r = ftruncate(fd, n_bytes);
  r_= ftruncate(fd_,n_bytes);
  if (r != 0 or r_ !=0) {
    perror("valkka_core: sharedmem.cpp: SharedMemSegment::serverInit: ftruncate failed");
    exit(2);
  }
  
  ptr = mmap(0, n_bytes,             PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ptr_= mmap(0, sizeof(std::size_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  
  if (ptr == MAP_FAILED or ptr_ == MAP_FAILED) {
    perror("valkka_core: sharedmem.cpp: SharedMemSegment::serverInit: mmap failed");
    exit(2);
  }
  
  payload = (uint8_t*) ptr;
  meta    = (std::size_t*) ptr_;
  
  close(fd);
  close(fd_);
}


void SharedMemSegment::serverClose() {
 if (munmap(ptr, n_bytes)!=0 or munmap(ptr_, sizeof(std::size_t)!=0)) {
    perror("valkka_core: sharedmem.cpp: SharedMemSegment::serverClose: munmap failed");
    exit(2);
  }
  
  if (shm_unlink(payload_name.c_str())!=0 or shm_unlink(meta_name.c_str())!=0) {
    perror("valkka_core: sharedmem.cpp: SharedMemSegment::serverClose: shm_unlink failed");
    exit(2);
  }
} 

    
bool SharedMemSegment::clientInit() {
  int fd, fd_;
  
  fd = shm_open(payload_name.c_str(), O_RDONLY, 0400);
  fd_= shm_open(meta_name.c_str(), O_RDONLY, 0400);
  
  if (fd == -1 or fd_==-1) {
    // perror("valkka_core: sharedmem.cpp: SharedMemSegment::clientInit: shm_open failed");
    // exit(2);
    return false;
  }
  
  // std::cout << "got shmem" << std::endl;
  ptr  =mmap(0, n_bytes, PROT_READ, MAP_SHARED, fd,  0);
  ptr_ =mmap(0, sizeof(std::size_t), PROT_READ, MAP_SHARED, fd_, 0);
  
  if (ptr == MAP_FAILED or ptr_ == MAP_FAILED) {
    std::cout << "valkka_core: sharedmem.cpp: SharedMemSegment::clientInit: mmap failed" << std::endl;
    // perror("valkka_core: sharedmem.cpp: SharedMemSegment::clientInit: mmap failed");
    // exit(2);
    return false;
  }
  
  payload = (uint8_t*) ptr;
  meta    = (std::size_t*) ptr_;
  
  close(fd);
  close(fd_);
  
  return true;
}


void SharedMemSegment::clientClose() {
}
  
    
void SharedMemSegment::put(std::vector<uint8_t> &inp_payload) {
  *meta = std::min(inp_payload.size(),n_bytes);
  // std::cout << "SharedMemSegment: put: inp_payload: " << int(inp_payload[0]) << " " << int(inp_payload[1]) << " " << int(inp_payload[2]) << std::endl;
  // std::cout << "SharedMemSegment: put: meta=" << *meta << std::endl;
  memcpy(payload, inp_payload.data(), *meta);
  // std::cout << "SharedMemSegment: put: payload now: " << int(payload[0]) << " " << int(payload[1]) << " " << int(payload[2]) << std::endl;
}


void SharedMemSegment::putAVRGBFrame(AVRGBFrame *f) { // copy from AVFrame->data directly
  // FFmpeg libav rgb has everything in av_frame->data[0].  The number of bytes is av_frame->linesize[0]*av_frame->height.
  AVFrame *av_frame =f->av_frame;
  
  // number of bytes:
  *meta =std::min(std::size_t(av_frame->linesize[0]*av_frame->height), n_bytes);
  
  // std::cout << "SharedMemSegment: putAVRGBFrame: copying " << *meta << " bytes from " << *f << std::endl;
  memcpy(payload, av_frame->data[0], *meta);
}


std::size_t SharedMemSegment::getSize() {
  return *meta;
}

bool SharedMemSegment::getClientState() {
  return client_state;
}



SharedMemRingBuffer::SharedMemRingBuffer(const char* name, int n_cells, std::size_t n_bytes, int mstimeout, bool is_server) : name(name), n_cells(n_cells), n_bytes(n_bytes), mstimeout(mstimeout), is_server(is_server) {
  int i;
  
  // std::cout << "SharedMemRingBuffer: constructor" << std::endl;
  
  sema_name    =std::string("/")+name+std::string("_valkka_ringbuffer");
  flagsema_name=std::string("/")+name+std::string("_valkka_ringflag");
  
  for(i=0; i<n_cells; i++) {
    shmems.push_back(
      new SharedMemSegment((name+std::to_string(i)).c_str(), n_bytes, is_server)
           );
  }
  
  sema     =sem_open(sema_name.c_str(),    O_CREAT,0600,0);
  flagsema =sem_open(flagsema_name.c_str(),O_CREAT,0600,0);
  
  index=-1;
    
  if (is_server) {
    zero();
    clearFlag();
  }
  
}


SharedMemRingBuffer::~SharedMemRingBuffer() {
  for (auto it=shmems.begin(); it!=shmems.end(); ++it) {
    delete *it;
  }
  sem_close(sema);
  sem_close(flagsema);
}
  

void  SharedMemRingBuffer::setFlag() {       //Server: call this to indicate a ring-buffer overflow
  int i;
  if (getFlagValue()>0) { // flag already set ..
    return;
  }
  else { // flag not set, increment semaphore
    i=sem_post(flagsema);
  }  
}


bool  SharedMemRingBuffer::flagIsSet() {     //Client: call this to see if there has been a ring-buffer overflow
  if (this->getFlagValue()>0) {
    return true;
  }
  else {
    return false;
  }
}
  

void  SharedMemRingBuffer::clearFlag() {    //Client: call this after handling the ring-buffer overflow
  int i;
  if (this->getFlagValue()>0) { // flag is set 
    i=sem_wait(flagsema); // ..not anymore: this decrements it immediately
  }
  else {
    return;
  }
}
  

int   SharedMemRingBuffer::getFlagValue() {  //Used by SharedMemoryRingBuffer::flagIsSet()
  int i;
  int semaval;
  i=sem_getvalue(flagsema, &semaval);
  return semaval;  
}


void  SharedMemRingBuffer::zero() {          //Force reset.  Semaphore value is set to 0
  // std::cout << "RingBuffer: zero"<<std::endl;
  int i;
  // while (errno!=EAGAIN) {
  while (getValue()>0) {
    i=sem_trywait(sema);
  }
}


int   SharedMemRingBuffer::getValue() {      //Returns the current index (next to be read) of the shmem buffer
  int i;
  int semaval;
  i=sem_getvalue(sema, &semaval);
  return semaval;
}  


bool SharedMemRingBuffer::getClientState() {
  int i;
  bool ok;
  
  ok=true;
  for(i=0; i<n_cells; i++) {
    ok = (ok and shmems[i]->getClientState());
  }
  
  return ok;
}


void SharedMemRingBuffer::serverPush(std::vector<uint8_t> &inp_payload) {
  int i;
    
  if (getValue()>=n_cells) { // so, semaphore will overflow
    zero();
    std::cout << "RingBuffer: ServerPush: zeroed, value now="<<getValue()<<std::endl;
    index=-1;
    setFlag();
    std::cout << "RingBuffer: ServerPush: OVERFLOW "<<std::endl;
  }
  
  ++index;
  if (index>=n_cells) {
    index=0;
  }
  shmems[index]->put(inp_payload);
  // std::cout << "RingBuffer: ServerPush: wrote to index "<<index<<std::endl;
  
  i=sem_post(sema);
}



bool SharedMemRingBuffer::clientPull(int &index_out, int &size_out) {
  int i;
  
  index_out =0;
  size_out  =0;
  
  if (mstimeout==0) {
    while ((i = sem_wait(sema)) == -1 && errno == EINTR)
      continue; // Restart if interrupted by handler
  }
  else {
    if (clock_gettime(CLOCK_REALTIME, &ts) == -1)
    {
      perror("SharedMemRingBuffer: clientPull: clock_gettime failed");
      exit(2);
    }
    
    // void set_normalized_timespec(struct timespec *ts, time_t sec, s64 nsec)
    // std::cout << "SharedMemoryRingBuffer: clientPull: using "<< mstimeout <<" ms timeout" << std::endl;
    // std::cout << "SharedMemoryRingBuffer: clientPull: time0 " << ts.tv_sec << " " << ts.tv_nsec << std::endl;
    normalize_timespec(&ts, ts.tv_sec, ts.tv_nsec+(int64_t(mstimeout)*1000000));
    /*
    ts.tv_nsec+=(mstimeout*1000000);
    ts.tv_sec +=ts.tv_nsec/1000000000; // integer division
    ts.tv_nsec =ts.tv_nsec%1000000000;
    */
    // std::cout << "SharedMemoryRingBuffer: clientPull: time1 " << ts.tv_sec << " " << ts.tv_nsec << std::endl;
    
    // i=sem_timedwait(sema, &ts);
    while ((i = sem_timedwait(sema, &ts)) == -1 && errno == EINTR)
      continue; // Restart if interrupted by handler
  }
  
  /* Check what happened */
  if (i == -1)
  {
    if (errno == ETIMEDOUT) {
      // printf("sem_timedwait() timed out\n");
      return false;
    }
    else 
    {
      perror("SharedMemRingBuffer: clientPull: sem_timedwait failed");
      exit(2);
    } 
  }
    
  if (flagIsSet()) {
    index=-1;
    clearFlag();
  }
  ++index;
  if (index>=n_cells) {
    index=0;
  }
  // TODO: read data here // what if read/write at the same moment at same cell..?  is locking a better idea ..? nope .. no locking needed!   This is process-safe by architecture..
  // std::cout << "RingBuffer: clientPull: read index "<<index<<std::endl;
  
  index_out=index;
  size_out =shmems[index]->getSize();
  // return index;
  return true;
}



SharedMemRingBufferRGB::SharedMemRingBufferRGB(const char* name, int n_cells, int width, int height, int mstimeout, bool is_server) : SharedMemRingBuffer(name, n_cells, (std::size_t)(width*height*3), mstimeout, is_server) {
}


SharedMemRingBufferRGB::~SharedMemRingBufferRGB() {
}


void SharedMemRingBufferRGB::serverPushAVRGBFrame(AVRGBFrame *f) {
  int i;
    
  if (getValue()>=n_cells) { // so, semaphore will overflow
    zero();
    std::cout << "RingBuffer: ServerPush: zeroed, value now="<<getValue()<<std::endl;
    index=-1;
    setFlag();
    std::cout << "RingBuffer: ServerPush: OVERFLOW "<<std::endl;
  }
  
  ++index;
  if (index>=n_cells) {
    index=0;
  }
  shmems[index]->putAVRGBFrame(f);
  // std::cout << "RingBuffer: ServerPush: wrote to index "<<index<<std::endl;
  
  i=sem_post(sema);
}



ShmemFrameFilter::ShmemFrameFilter(const char* name, int n_cells, std::size_t n_bytes, int mstimeout) : FrameFilter(name,NULL), shmembuf(name, n_cells, n_bytes, mstimeout, true) {
}
  

void ShmemFrameFilter::go(Frame* frame) {
  // shmembuf.serverPush(frame->payload); // this depends on the FrameClass this shmemfilter is supposed to manipulate
  if (frame->getFrameClass()!=FrameClass::basic) {
    std::cout << "ShmemFrameFilter: go: ERROR: BasicFrame required" << std::endl;
    return;
  }
  BasicFrame *basic =static_cast<BasicFrame*>(frame);
  shmembuf.serverPush(basic->payload);
}


//RGBShmemFrameFilter::RGBShmemFrameFilter(const char* name, int n_cells, int width, int height, int mstimeout) : ShmemFrameFilter(name, n_cells, width*height*3, mstimeout) {
//}
RGBShmemFrameFilter::RGBShmemFrameFilter(const char* name, int n_cells, int width, int height, int mstimeout) : FrameFilter(name,NULL), shmembuf(name, n_cells, width, height, mstimeout, true) {
}


void RGBShmemFrameFilter::go(Frame* frame) {
  if (frame->getFrameClass()!=FrameClass::avrgb) {
    std::cout << "BitmapShmemFrameFilter: go: ERROR: AVBitmapFrame required" << std::endl;
    return;
  }
  AVRGBFrame *rgbframe =static_cast<AVRGBFrame*>(frame);
  
  shmembuf.serverPushAVRGBFrame(rgbframe);
}



