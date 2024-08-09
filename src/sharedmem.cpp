/*
 * sharedmem.cpp : Posix shared memory segment server/client management, shared memory ring buffer synchronized using posix semaphores.
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
 *  @file    sharedmem.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.6.1 
 *  
 *  @brief   Posix shared memory segment server/client management, shared memory ring buffer synchronized using posix semaphores.
 */ 

#include <poll.h>
#include "tools.h"
#include "sharedmem.h"
#include "numpy_no_import.h"

// #define shmem_verbose


EventFd::EventFd() {
    this->fd = eventfd(0, EFD_NONBLOCK); // https://linux.die.net/man/2/eventfd
    if (fd == -1) {
        handle_error("SharedMemRingBufferBase: ctor: could not get eventfd handle");
    }
}

EventFd::~EventFd() {
    close(this->fd);
}

int EventFd::getFd() {
    return fd;
}

void EventFd::set() {
    uint64_t u = 1; // one frame
    ssize_t s;
    s = write(fd, &u, sizeof(uint64_t));
    if (s != sizeof(uint64_t)) {
        handle_error("EventFd: write failed");
    }
}

void EventFd::clear() {
    uint64_t u = 1; // one frame
    ssize_t s;
    s = read(fd, &u, sizeof(uint64_t));
    if (s != sizeof(uint64_t)) {
        handle_error("EventFd: read failed");
    }    
}


SharedMemSegment::SharedMemSegment(const char* name, std::size_t n_bytes, bool is_server) : name(name), n_bytes(n_bytes), is_server(is_server), client_state(false) {
  payload_name=std::string("/")+name+std::string("_valkka_payload");
  meta_name   =std::string("/")+name+std::string("_valkka_meta");
}


void SharedMemSegment::init() {
    if (is_server) {
        #ifdef shmem_verbose
        std::cout << "SharedMemSegment::init : server" << std::endl;
        #endif
        serverInit();
    }
    else {
        #ifdef shmem_verbose
        std::cout << "SharedMemSegment::init : client" << std::endl;
        #endif
        client_state = clientInit();
    }
}


void SharedMemSegment::close_() { 
    if (is_server) {
        // std::cout << "close server shmem segment " << name << std::endl;
        serverClose();
    }
    else {
        // std::cout << "close client shmem segment " << name << std::endl;
        clientClose();
    }
}


SharedMemSegment::~SharedMemSegment() {
    // std::cout << "SharedMemSegment: dtor" << std::endl;
}


bool SharedMemSegment::getClientState() {
  return client_state;
}



/*
META SharedMemSegment::getMeta() {
    return *meta;
}
*/

  
SimpleSharedMemSegment::SimpleSharedMemSegment(const char* name, std::size_t n_bytes, bool is_server) : SharedMemSegment(name, n_bytes, is_server) {
}


SimpleSharedMemSegment::~SimpleSharedMemSegment() {
}



server_init(SimpleSharedMemSegment, std::size_t);
client_init(SimpleSharedMemSegment, std::size_t);
server_close(SimpleSharedMemSegment, std::size_t);
client_close(SimpleSharedMemSegment, std::size_t);
copy_meta_from(SimpleSharedMemSegment, std::size_t);
copy_meta_to(SimpleSharedMemSegment, std::size_t);


std::size_t SimpleSharedMemSegment::getSize() {
  return *meta;
}


void SimpleSharedMemSegment::put(std::vector<uint8_t> &inp_payload, void* meta_) {
    // *meta = *((std::size_t*)(meta_));
    *meta = std::min(*((std::size_t*)(meta_)), n_bytes); // correct size aka metadata
    memcpy(payload, inp_payload.data(), *meta);
    // std::cout << "SharedMemSegment: put: payload now: " << int(payload[0]) << " " << int(payload[1]) << " " << int(payload[2]) << std::endl;
}

void SimpleSharedMemSegment::put(uint8_t* buf, void* meta_) {
    // *meta = *((std::size_t*)(meta_));
    *meta = std::min(*((std::size_t*)(meta_)), n_bytes); // correct size aka metadata
    memcpy(payload, buf, *meta);
    // std::cout << "SharedMemSegment: put: payload now: " << int(payload[0]) << " " << int(payload[1]) << " " << int(payload[2]) << std::endl;
}   
    
void SimpleSharedMemSegment::put(std::vector<uint8_t> &inp_payload) {
    std::size_t size = std::min(inp_payload.size(), n_bytes);
    std::cout << "size " << size << std::endl;
    this->put(inp_payload, (void*)(&size));
}
    
    
void SimpleSharedMemSegment::putAVRGBFrame(AVRGBFrame *f) { // copy from AVFrame->data directly
  // FFmpeg libav rgb has everything in av_frame->data[0].  The number of bytes is av_frame->linesize[0]*av_frame->height.
  AVFrame *av_frame = f->av_frame;
  
  // number of bytes:
  *meta = std::min(std::size_t(av_frame->linesize[0]*av_frame->height), n_bytes);
  
  // std::cout << "SharedMemSegment: putAVRGBFrame: copying " << *meta << " bytes from " << *f << std::endl;
  memcpy(payload, av_frame->data[0], *meta);
}


RGB24SharedMemSegment::RGB24SharedMemSegment(const char* name, int width, int height, bool is_server) : SharedMemSegment(name, std::size_t(width*height*3), is_server) {
}


RGB24SharedMemSegment::~RGB24SharedMemSegment() {
    // std::cout << "RGB24SharedMemSegment: dtor" << std::endl;
}

server_init(RGB24SharedMemSegment, RGB24Meta);
client_init(RGB24SharedMemSegment, RGB24Meta);
server_close(RGB24SharedMemSegment, RGB24Meta);
client_close(RGB24SharedMemSegment, RGB24Meta);
copy_meta_from(RGB24SharedMemSegment, RGB24Meta);
copy_meta_to(RGB24SharedMemSegment, RGB24Meta);



void RGB24SharedMemSegment::put(std::vector<uint8_t> &inp_payload, void* meta_) {
    *meta = *((RGB24Meta*)(meta_));
    meta->size = std::min(std::size_t(meta->width*meta->height*3), n_bytes);
    memcpy(payload, inp_payload.data(), meta->size);
}

void RGB24SharedMemSegment::put(uint8_t* buf, void* meta_) {
    *meta = *((RGB24Meta*)(meta_)); // this copies metadata to this sharedmem segment
    meta->size = std::min(std::size_t(meta->width*meta->height*3), n_bytes);
    // check that serialization worked by de-serializing:
    /*
    std::cout << ">>>put : "
        << " size, w, h, slot: " << meta->size << " " << meta->width << " " << meta->height << " " << meta->slot 
        << std::endl;
    */
    memcpy(payload, buf, meta->size);
    // memcpy(payload, buf, 1); // debug/test
}


void RGB24SharedMemSegment::putAVRGBFrame(AVRGBFrame *f) { // copy from AVFrame->data directly    
    // FFmpeg libav rgb has everything in av_frame->data[0].  The number of bytes is av_frame->linesize[0]*av_frame->height.
    AVFrame *av_frame = f->av_frame;
    
    RGB24Meta meta_ = RGB24Meta();
    
    meta_.width = av_frame->width;
    meta_.height = av_frame->height;
    meta_.slot = f->n_slot;
    meta_.mstimestamp = f->mstimestamp;
    *meta = meta_; 
    meta->size = std::min(std::size_t(meta_.width*meta_.height*3), n_bytes);
    
    // std::cout << "RGB24SharedMemSegment: putAVRGBFrame: copying " << meta->size << " bytes from " << *f << std::endl;
    memcpy(payload, av_frame->data[0], meta->size);
}


std::size_t RGB24SharedMemSegment::getSize() {
  return std::size_t(meta->width*meta->height*3);
}




SharedMemRingBufferBase::SharedMemRingBufferBase(const char* name, int n_cells, 
    std::size_t n_bytes, int mstimeout, bool is_server) : 
    name(name), n_cells(n_cells), n_bytes(n_bytes), 
    mstimeout(mstimeout), is_server(is_server), fd(0) {
    int i;

    // std::cout << "SharedMemRingBufferBase: constructor: n_cells " << n_cells << std::endl;

    sema_name    =std::string("/")+name+std::string("_valkka_ringbuffer");
    flagsema_name=std::string("/")+name+std::string("_valkka_ringflag");

    /*
    for(i=0; i<n_cells; i++) {
        shmems.push_back(
        new SEGMENT((name+std::to_string(i)).c_str(), n_bytes, is_server)
            );
    }
    */
    sema     =sem_open(sema_name.c_str(),    O_CREAT,0600,0);
    flagsema =sem_open(flagsema_name.c_str(),O_CREAT,0600,0);

    index=-1;

    if (is_server) {
        zero();
        clearFlag();
    }
    else{
        #ifdef USE_SHMEM_CACHE
        this->cache = new uint8_t*[n_cells];
        for(i=0; i<n_cells; i++){
            this->cache[i] = new uint8_t[n_bytes];
        }
        #endif
    }
}

/*
- Server: create eventfd: .serverCreateFd() => returns file descriptor number
- Client: use eventfd: .clientUseFd()
- ..after that, client can poll the eventfd
*/


SharedMemRingBufferBase::~SharedMemRingBufferBase() {
    // std::cout << "SharedMemRigBufferBase: closing sema" << std::endl;
    sem_close(sema);
    sem_close(flagsema);
    if (is_server) {
        sem_unlink(sema_name.c_str());
        sem_unlink(flagsema_name.c_str());
    }
    else {
        #ifdef USE_SHMEM_CACHE
        int i;
        for(i=0; i<n_cells; i++) {
            delete[] this->cache[i];
        }
        delete[] this->cache;
        #endif
    }
    /* // noooooo!
    if (fd > 0) {
        close(this->fd);
    }
    */
}


void SharedMemRingBufferBase::serverUseFd(EventFd &event_fd) {
    fd = event_fd.getFd();
    if (fcntl(fd, F_GETFL) < 0 && errno == EBADF) {
        handle_error("RingBuffer: serverUseFd: invalid/closed fd");
    }
    /*
    // fcntl(fd, F_SETFL|O_NONBLOCK);
    struct pollfd fds[1];
    fds[0] = (struct pollfd){fd, POLLIN, 0};
    //int ppoll(struct pollfd *fds, nfds_t nfds,
    //           const struct timespec *tmo_p, const sigset_t *sigmask);
    // len of pollfd array, ms to wait
    int ret = poll(fds, 1, 0);
    std::cout << ">>serverUseFd : fd=" << fd << std::endl;
    std::cout << ">>serverUseFd : " << ret << std::endl;
    if (ret > 0) { // "reset" this eventfd
        std::cout << ">>serverUseFd : resetting eventfd " << std::endl;
        uint64_t u = 1; // one frame
        ssize_t s;
        s = read(fd, &u, sizeof(uint64_t));
        std::cout <<  ">>serverUseFd : s=" << s << std::endl;
        if (s != sizeof(uint64_t)) {
            handle_error("RingBuffer: serverUseFd: eventfd read failed");
        }
    }
    */
}

void SharedMemRingBufferBase::clientUseFd(EventFd &event_fd) {
    fd = event_fd.getFd();
}

void SharedMemRingBufferBase::setEventFd() { // used only at the server side
    //call after se_post
    //fd_is_set = false; // TODO: poll for the eventfd
    //if (fd > 0 and !fd_is_set) {
    // std::cout << ">> fd, getValue " << fd << " " << getValue() << std::endl;
    if (fd > 0 and getValue() >= 1) {
        // TODO?: write only if the eventfd is not set
        // TODO?: sync all semaphore operations with the fd
        uint64_t u = 1; // one frame
        ssize_t s;
        // std::cout << "setting eventfd" << std::endl;
        s = write(fd, &u, sizeof(uint64_t));
        if (s != sizeof(uint64_t)) {
            handle_error("RingBuffer: ServerPush: eventfd write failed");
        }
    }
}


void SharedMemRingBufferBase::clearEventFd() { // used only at the client
    if (fd > 0) {
        if (getValue() > 0) { // clear eventfd if no more frames in the queue
            return;
        }
        uint64_t u = 1; // one frame
        ssize_t s;
        s = read(fd, &u, sizeof(uint64_t));
        if (s != sizeof(uint64_t)) {
            handle_error("RingBuffer: clearEventFd: eventfd read failed");
        }
    }
}


void  SharedMemRingBufferBase::setFlag() {       //Server: call this to indicate a ring-buffer overflow
    int i;
    if (getFlagValue()>0) { // flag already set ..
        return;
    }
    else { // flag not set, increment semaphore
        i=sem_post(flagsema);
    }  
}


bool  SharedMemRingBufferBase::flagIsSet() {     //Client: call this to see if there has been a ring-buffer overflow
    if (this->getFlagValue()>0) {
        return true;
    }
    else {
        return false;
    }
}


void  SharedMemRingBufferBase::clearFlag() {    //Client: call this after handling the ring-buffer overflow
  int i;
  if (this->getFlagValue()>0) { // flag is set 
    i=sem_wait(flagsema); // ..not anymore: this decrements it immediately
  }
  else {
    return;
  }
}
  

int SharedMemRingBufferBase::getFlagValue() {  //Used by SharedMemoryRingBuffer::flagIsSet()
  int i;
  int semaval;
  i = sem_getvalue(flagsema, &semaval);
  return semaval;  
}


void  SharedMemRingBufferBase::zero() {          //Force reset.  Semaphore value is set to 0
  // std::cout << "RingBuffer: zero"<<std::endl;
  int i;
  // while (errno!=EAGAIN) {
  while (getValue()>0) {
    i=sem_trywait(sema);
  }
}


int   SharedMemRingBufferBase::getValue() {      //Returns the current index (next to be read) of the shmem buffer
  int i;
  int semaval;
  i=sem_getvalue(sema, &semaval);
  return semaval;
}  


bool SharedMemRingBufferBase::getClientState() {
  int i;
  bool ok;
  
  ok=true;
  for(i=0; i<n_cells; i++) {
    ok = (ok and shmems[i]->getClientState());
  }
  
  return ok;
}


void SharedMemRingBufferBase::serverPush(std::vector<uint8_t> &inp_payload, void *meta) {
    int i;
        
    if (getValue()>=n_cells) { // so, semaphore will overflow
        zero();
        std::cout << "RingBuffer " << name << " ServerPush: zeroed, value now="<<getValue()<<std::endl;
        index=-1;
        setFlag();
        std::cout << "RingBuffer " << name << " ServerPush: OVERFLOW "<<std::endl;
    }
    
    ++index;
    if (index>=n_cells) {
        index=0;
    }
    #ifdef SAFE_TEST
    {
        std::unique_lock<std::mutex> lock(test_mutex);
    #endif
        shmems[index]->put(inp_payload, meta); // SharedMemSegment takes care of the correct typecast from void*
        // std::cout << "RingBuffer: ServerPush: wrote to index "<<index<<std::endl;
        // std::cout << "sema pointer:" << long(sema) << std::endl;
    #ifdef SAFE_TEST
    }
    #endif
    i=sem_post(sema);
    setEventFd();
}


PyObject *SharedMemRingBufferBase::getBufferListPy() {
    PyObject *plis, *pa;
    npy_intp dims[1];
    
    // plis = PyList_New(0);
    plis = PyList_New(shmems.size());

    // return plis;
    // std::cout << name << " cache adr at pylist " << long(cache[0]) << std::endl;
    int i = 0;
    for (auto it = shmems.begin(); it != shmems.end(); ++it) {
        dims[0] = (*it)->n_bytes;
        #ifdef USE_SHMEM_CACHE
        // (b): expose memory segments from cache instead:
        pa = PyArray_SimpleNewFromData(1, dims, NPY_UBYTE, (char*)(
                this->cache[i]
            )); // use cache instead
        //*/
        #else
        // (a) expose shared memory directly as a numpy array
        pa = PyArray_SimpleNewFromData(1, dims, NPY_UBYTE, (char*)((*it)->payload));
        #endif
        // PyList_Append(plis, pa); // keep reference of pa
        PyList_SetItem(plis, i, pa); // transfer ownership of pa: correct
        i++;
    }
    return plis;
}


bool SharedMemRingBufferBase::clientPull(int &index_out, void *meta_) {
    int i;
    
    index_out =0;
    // size_out  =0;
    
    // std::cout << "::clientPull: mstimeout " << mstimeout << std::endl;
    // TODO: if server feeds (RGB) frames that are much larger than what
    // defined for the client, sem_timedwait can get stuck - might be a memflow

    if (mstimeout==0) {
        while ((i = sem_wait(sema)) == -1 && errno == EINTR)
        continue; // Restart if interrupted by handler
    }
    else {
        if (clock_gettime(CLOCK_REALTIME, &ts) == -1)
        {
            perror("SharedMemRingBufferBase: clientPull: clock_gettime failed");
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
        //std::cout << "::clientPull: sem_timedwait " << std::endl;
        while ((i = sem_timedwait(sema, &ts)) == -1 && errno == EINTR) {
            continue; // Restart if interrupted by handler
        }
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
            perror("SharedMemRingBufferBase: clientPull: sem_timedwait failed");
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
    // TODO: read data here // what if read/write at the same moment at same cell..?  
    // is locking a better idea ..? nope .. no locking needed!   This is process-safe by architecture..
    // std::cout << "RingBuffer: clientPull: read index "<<index<<std::endl;
    
    index_out = index;
    // size_out  = shmems[index]->getSize();
    #ifdef SAFE_TEST
    {
        std::unique_lock<std::mutex> lock(test_mutex);
    #endif
        shmems[index]->copyMetaTo(meta_); // copy metadata from the shmem segment's internal memory into variable referenced by the meta_ pointer.  Subclassing takes care of the correct typecast.
    #ifdef SAFE_TEST
    }
    #endif
    // return index;
    // std::cout << "size " << shmems[index]->getSize() << std::endl;
    #ifdef USE_SHMEM_CACHE
    // (b): if cache is the one exposed as numpy array, copy data therein first
    memcpy(cache[index], shmems[index]->payload, shmems[index]->getSize());
    // std::cout << name << " cache adr " << long(cache[0]) << std::endl;
    /*
    int ii;
    for(ii=0; ii<10; ii++) {
        std::cout << int(cache[index][ii]) << " ";
    }
    std::cout << std::endl;
    */
    #else
    // (a): do nothing, if shmem is exposed as numpy array
    #endif
    clearEventFd(); // clears eventfd only if semaphore value is 0
    return true;
}


bool SharedMemRingBufferBase::clientPullThread(int &index_out, void *meta_) {
    bool val;
    // hmm..
    // Swig python object is created at python side
    // it is then passed to this method
    // but here we can't explicitly the reference counter of meta_
    // since it's not a python object (!)
    // but in any case, all that should be taken care by swig
    Py_BEGIN_ALLOW_THREADS
    val = clientPull(index_out, meta_);
    Py_END_ALLOW_THREADS
    return val;
}


SharedMemRingBuffer::SharedMemRingBuffer(const char* name, int n_cells, std::size_t n_bytes, int mstimeout, bool is_server) : SharedMemRingBufferBase(name, n_cells, n_bytes, mstimeout, is_server)  {
    int i;
    for(i=0; i < n_cells; i++) {
        shmems.push_back(
        new SimpleSharedMemSegment((name+std::to_string(i)).c_str(), n_bytes, is_server)
            );
    }
    for(auto it = shmems.begin(); it != shmems.end(); ++it) {
        (*it)->init(); // init reserves the shmem at the correct subclass
    }
}


SharedMemRingBuffer::~SharedMemRingBuffer() {
    for(auto it = shmems.begin(); it != shmems.end(); ++it) {
        (*it)->close_();
        delete *it;
    }
}


void SharedMemRingBuffer::serverPush(std::vector<uint8_t> &inp_payload) {
    std::size_t size = inp_payload.size();
    SharedMemRingBufferBase::serverPush(inp_payload, (void*)(&size));
}


bool SharedMemRingBuffer::serverPushPy(PyObject *po) {
    Py_INCREF(po);
    PyArrayObject *pa = (PyArrayObject*)po;
    if (PyArray_NDIM(pa) > 1) {
        std::cout 
            << "RingBuffer: ServerPushPy: incorrect dimensions: must be flat" 
            << std::endl;
        return false;
    }
    npy_intp *dims = PyArray_DIMS(pa);
    uint8_t* buf = (uint8_t*)PyArray_BYTES(pa);
    std::size_t size = (std::size_t)(dims[0]);

    // *****************
    int i;  
    if (getValue()>=n_cells) { // so, semaphore will overflow
        zero();
        std::cout << "RingBuffer " << name << " ServerPush: zeroed, value now="<<getValue()<<std::endl;
        index=-1;
        setFlag();
        std::cout << "RingBuffer " << name << " ServerPush: OVERFLOW "<<std::endl;
    }
    ++index;
    if (index>=n_cells) {
        index=0;
    }
    #ifdef SAFE_TEST
    {
        std::unique_lock<std::mutex> lock(test_mutex);
    #endif
        shmems[index]->put(buf, (void*)(&size)); // SharedMemSegment takes care of the correct typecast from void*
    #ifdef SAFE_TEST
    }
    #endif
    // std::cout << "RingBuffer: ServerPush: wrote to index "<<index<<std::endl;
    // std::cout << "sema pointer:" << long(sema) << std::endl;
    i=sem_post(sema);
    setEventFd();
    // ********************

    Py_DECREF(po);
    return true;
}


bool SharedMemRingBuffer::clientPull(int &index_out, int &size_out) {
    // std::size_t *size_out_ptr;
    return SharedMemRingBufferBase::clientPull(index_out, (void*)(&size_out));
}


SharedMemRingBufferRGB::SharedMemRingBufferRGB(const char* name, int n_cells, int width, int height, int mstimeout, bool is_server) : SharedMemRingBufferBase(name, n_cells, (std::size_t)(width*height*3), mstimeout, is_server), width(width), height(height) {
    int i;
    for(i=0; i<n_cells; i++) {
        #ifdef shmem_verbose
        std::cout << "create RGB24SharedMemSegment " << i << " is server =" << int(is_server) << std::endl;
        #endif
        shmems.push_back(
        new RGB24SharedMemSegment((name+std::to_string(i)).c_str(), width, height, is_server)
            );
    }
    for(auto it = shmems.begin(); it != shmems.end(); ++it) {
        #ifdef shmem_verbose
        std::cout << "call RGB24SharedMemSegment init " << std::endl;
        #endif
        (*it)->init(); // init reserves the shmem at the correct subclass
    }
}


SharedMemRingBufferRGB::~SharedMemRingBufferRGB() {
    // std::cout << "SharedMemRingBufferRGB: dtor" << std::endl;
    for(auto it = shmems.begin(); it != shmems.end(); ++it) {
        (*it)->close_();
        delete *it;
    }
}


void SharedMemRingBufferRGB::serverPushAVRGBFrame(AVRGBFrame *f) {
    int i;
        
    if (getValue()>=n_cells) { // so, semaphore will overflow
        zero();
        std::cout << "RingBuffer " << name << " ServerPush: zeroed, value now="<<getValue()<<std::endl;
        index=-1;
        setFlag();
        std::cout << "RingBuffer " << name << " ServerPush: OVERFLOW "<<std::endl;
    }
    
    ++index;
    if (index >= n_cells) {
        index=0;
    }

    #ifdef SAFE_TEST
    {
        std::unique_lock<std::mutex> lock(test_mutex);
    #endif
        ( (RGB24SharedMemSegment*)(shmems[index]) )->putAVRGBFrame(f);
    #ifdef SAFE_TEST
    }
    #endif
    //std::cout << "RingBuffer: ServerPush: wrote to index "<<index<<std::endl;
    
    i=sem_post(sema);
    setEventFd();
    //setEventFd(); // test: double-write ok
}


bool SharedMemRingBufferRGB::serverPushPyRGB(PyObject *po, SlotNumber slot, long int mstimestamp) {
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
        std::cout << "RingBuffer " << name << " ServerPush: zeroed, value now="<<getValue()<<std::endl;
        index=-1;
        setFlag();
        std::cout << "RingBuffer " << name << " ServerPush: OVERFLOW "<<std::endl;
    }
    
    ++index;
    if (index >= n_cells) {
        index=0;
    }

    /*
    std::cout << ">>>pushpyrgb: i, size, w, h, slot: " << index << " "
        << meta_.size << " " << meta_.width << " " << meta_.height << " " << meta_.slot 
        << std::endl;
    */
    
    #ifdef SAFE_TEST
    {
        std::unique_lock<std::mutex> lock(test_mutex);
    #endif
        ( (RGB24SharedMemSegment*)(shmems[index]) )->put(buf, (void*)&meta_);
    #ifdef SAFE_TEST
    }
    #endif
    // std::cout << "RingBuffer: ServerPush: wrote to index "<<index<<std::endl;
    
    Py_DECREF(po);

    i=sem_post(sema);
    setEventFd();
    return true;
}


bool SharedMemRingBufferRGB::clientPull(int &index_out, int &size_out) { // support for legacy code
    RGB24Meta meta_ = RGB24Meta();
    bool ok;    
    ok = SharedMemRingBufferBase::clientPull(index_out, (void*)(&meta_)); // writes to index_out and meta
    size_out = (int)(meta_.size);
    return ok;
}


bool SharedMemRingBufferRGB::clientPullFrame(int &index_out, RGB24Meta &meta_) {
    //bool ok;
    //ok = SharedMemRingBufferBase::clientPull(index_out, (void*)(&meta_));
    // std::cout << name << " --> pull: meta_.size = " << meta_.size << std::endl;
    return SharedMemRingBufferBase::clientPull(index_out, (void*)(&meta_));
    // return ok;
}


bool SharedMemRingBufferRGB::clientPullFrameThread(int &index_out, RGB24Meta &meta_) {
    return SharedMemRingBufferBase::clientPullThread(index_out, (void*)(&meta_));
}


PyObject* SharedMemRingBufferRGB::clientPullPy() {
    PyObject* tup = PyTuple_New(6);
    RGB24Meta meta = RGB24Meta();
    int index_out = 0;
    bool ok;
    // std::cout << ">>allow threads" << std::endl;
    Py_BEGIN_ALLOW_THREADS
    //std::cout << ">>clientPull" << std::endl;
    ok = SharedMemRingBufferBase::clientPull(index_out, (void*)(&meta));
    //std::cout << ">>clientPull end" << std::endl;
    Py_END_ALLOW_THREADS
    //std::cout << ">>end allow threads" << std::endl;

    /*
    std::cout << ">>>pull: index, size, width, height " << index_out << " " << meta.size
        << " " << meta.width
        << " " << meta.height
        << std::endl;
    */

    if (!ok) {
        index_out = -1;
        PyTuple_SetItem(tup, 0, PyLong_FromLong((long)index_out));
    }
    else {
        PyTuple_SetItem(tup, 0, PyLong_FromLong((long)index_out));
        PyTuple_SetItem(tup, 1, PyLong_FromSsize_t(meta.size));
        PyTuple_SetItem(tup, 2, PyLong_FromLong((long)meta.width));
        PyTuple_SetItem(tup, 3, PyLong_FromLong((long)meta.height));
        PyTuple_SetItem(tup, 4, PyLong_FromUnsignedLong((unsigned long)(meta.slot))); // unsigned short
        PyTuple_SetItem(tup, 5, PyLong_FromLong(meta.mstimestamp));
    }
    return tup;
}



ShmemFrameFilter::ShmemFrameFilter(const char* name, int n_cells, std::size_t n_bytes, int mstimeout) : FrameFilter(name,NULL), shmembuf(name, n_cells, n_bytes, mstimeout, true) {
}

void ShmemFrameFilter::useFd(EventFd &event_fd) {
    shmembuf.serverUseFd(event_fd);
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


void RGBShmemFrameFilter::useFd(EventFd &event_fd) {
    shmembuf.serverUseFd(event_fd);
}


void RGBShmemFrameFilter::go(Frame* frame) {
  if (frame->getFrameClass()!=FrameClass::avrgb) {
    std::cout << "RGBShmemFrameFilter: go: ERROR: AVBitmapFrame required" << std::endl;
    return;
  }
  AVRGBFrame *rgbframe =static_cast<AVRGBFrame*>(frame);
  
  shmembuf.serverPushAVRGBFrame(rgbframe);
}



