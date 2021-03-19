/*
 * cachestream.cpp :
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
 *  @file    cachestream.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.2.0 
 *  
 *  @brief 
 */ 

#include "cachestream.h"


CacheFrameFilter::CacheFrameFilter(const char* name, FrameCache* framecache) : FrameFilter(name), framecache(framecache) {
};

void CacheFrameFilter::go(Frame* frame) {
    // std::cout << "CacheFrameFilter : go : " << *frame << std::endl;
    framecache->writeCopy(frame);
}




FrameCache::FrameCache(const char *name, FrameCacheContext ctx) : name(name), ctx(ctx), mintime_(9999999999999), maxtime_(0), has_delta_frames(false) {
    state = cache.end();
}

FrameCache::~FrameCache() {
    clear();
}
 
bool FrameCache::writeCopy(Frame* f, bool wait) {
    std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
    // TODO: insert in order
    
    
    if (f->getFrameClass()==FrameClass::marker) {
        MarkerFrame *markerframe = static_cast<MarkerFrame*>(f);
        valkkafslogger.log(LogLevel::debug) << "FrameCache: writeCopy: got marker frame " << *markerframe << std::endl;
        if (markerframe->tm_start) {
            clear_();
        }
    }
            
    mintime_=std::min(mintime_, f->mstimestamp);
    maxtime_=std::max(maxtime_, f->mstimestamp);
    
    // std::cout << "FrameCache::writeCopy : mintime, maxtime, time " << mintime_ << " " << maxtime_ << " " << f->mstimestamp << std::endl;
    
    has_delta_frames = (has_delta_frames or !(f->isSeekable())); // frames that describe changes to previous frames are present

    // std::cout << "FrameCache: writeCopy: " << name << " : " << *f << std::endl;
    cache.push_back(f->getClone());  // push_back takes a copy of the pointer
    // dump_();

    #ifdef TIMING_VERBOSE
    long int dt=(getCurrentMsTimestamp()-f->mstimestamp);
    if (dt>100) {
        valkkafslogger.log(LogLevel::debug) << "FrameCache: "<<name<<" writeCopy : timing : inserting frame " << dt << " ms late" << std::endl;
    }
    #endif

    this->condition.notify_one(); // after receiving 
    return true;
}
 

/*
Frame* FrameCache::read(unsigned short int mstimeout) { // TODO: do we need this?
    return NULL;
}
*/

void FrameCache::clear_() {
    mintime_ = 9999999999999;
    maxtime_ = 0;
    has_delta_frames=false;
    for (auto it=cache.begin(); it!=cache.end(); ++it) {
        delete *it;
    }
    cache.clear(); // woops! don't forget this!  otherwise your manipulating a container with void pointers..
    state = cache.end();
    // std::cout << "FrameCache: clear: exit" << std::endl;
    // dump_();
}


void FrameCache::clear() {
    std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
    clear_();
}


void FrameCache::dump_() {
    std::cout << "FrameCache : dump : > " << name << " : " << std::endl;
    for (auto it=cache.begin(); it!=cache.end(); ++it) {
        // std::cout << "FrameCache : dump : " << name << " : " << **it << " [" << (*it)->dumpPayload() << std::endl;
        std::cout << "FrameCache : dump : " << name << " : " << **it;
        if ((*it)->isSeekable()) {
            std::cout << " * ";
        }
        std::cout << std::endl;
    }
    std::cout << "FrameCache : dump : < " << name << " : " << std::endl;
}



void FrameCache::dump() {
    std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
    dump_();
}

bool FrameCache::isEmpty() {
    std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
    return (cache.size() < 1);
}


int FrameCache::seek(long int ms_streamtime_) {
    std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
    // std::cout << "FrameCache : seek : mintime, maxtime, seek time " << mintime_ << " " << maxtime_ << " " << ms_streamtime_ << std::endl;
    
    state = cache.end();
    if (cache.size() < 1) {
        return -1;
    }
    if (ms_streamtime_ < mintime_) {
        return -1;
    }
    if (ms_streamtime_ > maxtime_) {
        return 1;
    }
    
    // backward iteration from the last frame
    Cache::reverse_iterator it; // iterate from right (old) to left (young)
    Frame *f; // shorthand
    
    f=NULL;
    for(auto it=cache.rbegin(); it!=cache.rend(); ++it) { // FRAME ITER
        f = *it;
        if (f->n_slot > 0) { // SLOT > 0
            if (f->mstimestamp <= ms_streamtime_) {
                break;
            }
        } // SLOT > 0
    } // FRAME ITER

    if (it == cache.rend()) { // left overflow
        return -1;
    }

    if (f) {
        std::cout << "FrameCache : seek : frame = " << *f;
    }
    
    // let's get the forward iterator
    Cache::iterator it2;
    for (it2=cache.begin(); it2!=cache.end(); ++it2) {
        if (f==*it2) {break;}
    }
    
    state = it2;
    /*
    f = *state;
    if (f) {
        std::cout << "FrameCache : state = " << *f;
    }
    */
    
    return 0; // ok!
}


int FrameCache::keySeek(long int ms_streamtime_) { // return values: -1 = no frames at left, 1 = no frames at right, 0 = ok
    std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
        
    valkkafslogger.log(LogLevel::debug) << "FrameCache : keySeek : mintime, maxtime, seek time " << mintime_ << " " << maxtime_ << " " << ms_streamtime_ << std::endl;
    
    state = cache.end();
    if (cache.size() < 1) {
        valkkafslogger.log(LogLevel::normal) << "FrameCache : keySeek : ERROR: cache.size" << std::endl;
        return -1;
    }
    if (ms_streamtime_ < mintime_) {
        valkkafslogger.log(LogLevel::normal) << "FrameCache : keySeek : ERROR: mintime_" << std::endl;
        return -1;
    }
    if (ms_streamtime_ > maxtime_) {
        valkkafslogger.log(LogLevel::normal) << "FrameCache : keySeek : ERROR: maxtime_" << std::endl;
        return 1;
    }
    
    // backward iteration from the last frame to the first necessary key-frame    
    std::map<SlotNumber, bool> req_map; // first: slot number.  second: has seek frame for this slot number been found
    Cache::reverse_iterator it; // iterate from right (old) to left (young)
    Frame *f; // shorthand
    
    f=NULL;
    for(auto it=cache.rbegin(); it!=cache.rend(); ++it) { // FRAME ITER
        f = *it;
        if (f->n_slot > 0) { // SLOT > 0
            // std::cout << "FrameCache: timestamp, target " << f->mstimestamp << ", " << ms_streamtime_ << std::endl;
            if (req_map.find(f->n_slot)==req_map.end()) { 
                // frames of this slot are present .. so a seek frame for this slot must be found
                // std::cout << "FrameCache: new slot found : " << f->n_slot << std::endl;
                req_map.insert(std::pair<SlotNumber, bool>(f->n_slot, false));
            }
            if (f->mstimestamp <= ms_streamtime_) {
                if (!has_delta_frames or f->isSeekable()) { // key-frame based or not
                    // std::cout << "FrameCache: seek ok'd slot " << f->n_slot << std::endl;
                    req_map[f->n_slot]=true; // found key-frame for this slot number (or any frame if key-frames not required)
                }
            }
            
            std::map<SlotNumber, bool>::iterator key_it;
            for (key_it=req_map.begin(); key_it!=req_map.end(); ++key_it) {
                if (!(key_it->second)) {
                    // must continue the search
                    break;
                }
            }
            
            if (key_it==req_map.end()) {
                // end was reached = seek frames for all relevant slots have been found
                break; // break FRAME ITER
            }
                
        } // SLOT > 0
    } // FRAME ITER

    
    if (it == cache.rend()) { // left overflow
        return -1;
    }

    if (f) {
        valkkafslogger.log(LogLevel::debug) << "FrameCache : seek : frame = " << *f;
    }
    
    // let's get the forward iterator
    Cache::iterator it2;
    for (it2=cache.begin(); it2!=cache.end(); ++it2) {
        if (f==*it2) {break;}
    }
    
    state = it2;
    /*
    f = *state;
    if (f) {
        std::cout << "FrameCache : state = " << *f;
    }
    */
    
    return 0; // ok!
}


Frame* FrameCache::pullNextFrame() {
    std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
    Frame* f;
    if (state == cache.end()) {
        return NULL;
    }
    f = *state;
    state++;
    return f;
}



FileCacheThread::FileCacheThread(const char *name) : AbstractFileThread(name), 
    frame_cache_1("cache_1"), frame_cache_2("cache_2"),     // define the filterchain from end to beginning
    cache_filter_1(name, &frame_cache_1), cache_filter_2(name, &frame_cache_2),
    switchfilter(name, &cache_filter_1, &cache_filter_2),   // by default, frames are going into frame_cache_1
    typefilter(name, FrameClass::marker, &infilter),        // infilter comes from the mother class
    fork(name, &switchfilter, &typefilter),
    play_cache(&frame_cache_2),
    tmp_cache(&frame_cache_1),
    callback(NULL), target_mstimestamp_(0), pyfunc(NULL), pyfunc2(NULL), next(NULL), reftime(0), walltime(0), state(AbstractFileState::stop)
    {
    this->slots_.resize(I_MAX_SLOTS+1,NULL);
    this->setup_frames.resize(I_MAX_SLOTS+1);
    for (auto it_slot = setup_frames.begin(); it_slot != setup_frames.end(); ++it_slot) {
        it_slot->resize(I_MAX_SUBSESSIONS+1, NULL);
    }
    // SetupFrame *f = setup_frames[255][4]; // overflow
    // SetupFrame *f = setup_frames[4][255]; // correct order of indexes is this // nopes ..
}


FileCacheThread::~FileCacheThread() {
    for (auto it_slot=setup_frames.begin(); it_slot!=setup_frames.end(); ++it_slot) {
        for (auto it_subs=it_slot->begin(); it_subs!=it_slot->end(); ++it_subs) {
            if (*it_subs) {
                delete *it_subs;
            }
        }
    } // yes, I know, this sucks
}

void FileCacheThread::setCallback(void func(long int)) {
    callback = func;
}

void FileCacheThread::setPyCallback(PyObject* pobj) {
    /* this is called from the main python process, not from a cpp thread, so no need
    for this..!
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    */
    // pass here, say "signal.emit" or a function/method that accepts single argument
    if (PyCallable_Check(pobj)) { // https://docs.python.org/3/c-api/type.html#c.PyTypeObject
        Py_INCREF(pobj);
        pyfunc=pobj;
    }
    else {
        valkkafslogger.log(LogLevel::fatal) << "FileCacheThread: setPyCallback: needs python callable for emitting current time" << std::endl;
        pyfunc=NULL;
    }
    
    // PyGILState_Release(gstate);
}

void FileCacheThread::setPyCallback2(PyObject* pobj) {
    /*
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    */
    
    // pass here, say "signal.emit" or a function/method that accepts single argument
    if (PyCallable_Check(pobj)) { // https://docs.python.org/3/c-api/type.html#c.PyTypeObject
        Py_INCREF(pobj);
        pyfunc2=pobj;
    }
    else {
        valkkafslogger.log(LogLevel::fatal) << "FileCacheThread: setPyCallback2: needs python callable for emitting loaded frame time limits" << std::endl;
        pyfunc2=NULL;
    }
    // PyGILState_Release(gstate);
}



void FileCacheThread::switchCache() {
    if (play_cache==&frame_cache_1) {
        play_cache=&frame_cache_2; // this thread is manipulating frame_cache_2
        tmp_cache=&frame_cache_1;
        switchfilter.set1();       // new frames are cached to frame_cache_1
        frame_cache_1.clear();     // .. but before that, clear it
    }
    else { // vice versa
        play_cache=&frame_cache_1;
        tmp_cache=&frame_cache_2;
        switchfilter.set2();
        frame_cache_2.clear();
    }
    
    if (pyfunc2) {
        valkkafslogger.log(LogLevel::debug) << "FileCacheThread: switchCache : evoke python callback : loaded frame limits" << std::endl;
        
        // This is not needed: PyObject_CallFunctionObjArgs does it ?
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
        
        PyObject *res, *tup;
    
        // res = Py_BuildValue("ll", (long int)(1), (long int)(2));
        // res = Py_BuildValue("(ii)", 123, 456); // debug : crassssh
        
        tup = PyTuple_New(2);
        PyTuple_SET_ITEM(tup, 0, PyLong_FromLong(play_cache->getMinTime_()));
        PyTuple_SET_ITEM(tup, 1, PyLong_FromLong(play_cache->getMaxTime_()));
        
        /*
        if (!res) {
            std::cout << "FileCacheThread: switchCache: WARNING: could not create tuple" << std::endl;
        }
        */
        
        // res = PyObject_CallFunctionObjArgs(pyfunc2, res);
        res = PyObject_CallFunctionObjArgs(pyfunc2, tup, NULL);
        
        // send a tuple with min and max loaded frame time values
        // res=PyObject_CallFunction(pyfunc2, "ll", play_cache->getMinTime_(), play_cache->getMaxTime_());  // nooo fucking way
        // res=PyObject_CallFunction(pyfunc2, "l", 0); // debug // works ok
        
        if (!res) {
            valkkafslogger.log(LogLevel::fatal) << "FileCacheThread: switchCache: WARNING: python callback failed : loaded frame limits" << std::endl;
        }
        PyGILState_Release(gstate);
    }
}
    
void FileCacheThread::dumpPlayCache() {
}

void FileCacheThread::dumpTmpCache() {
}


void FileCacheThread::sendSetupFrames(SetupFrame *f) {
    SlotNumber cc = 0;
    valkkafslogger.log(LogLevel::debug) << "FileCacheThread : sendSetupFrames : " << std::endl;
    for (auto it=slots_.begin(); it!=slots_.end(); ++it) {
        if (*it) {
            #ifdef CACHESTREAM_VERBOSE
            std::cout << "FileCacheThread : sendSetupFrames : " << cc << " : " << *f << std::endl;
            #endif
            f->n_slot = cc;
            (*it)->run(f);
        }
        cc++;
    }
}


void FileCacheThread::stopStreams(bool send_state) {
    if (reftime > 0) {
        state = AbstractFileState::stop;
        walltime = getCurrentMsTimestamp();
        target_mstimestamp_ = walltime - reftime;
        
        if (send_state) {
            state_setupframe.mstimestamp = walltime;
            state_setupframe.sub_type = SetupFrameType::stream_state;
            state_setupframe.stream_state = state;
            sendSetupFrames(&state_setupframe);
        }
    }
}

void FileCacheThread::playStreams(bool send_state) {
    if (target_mstimestamp_ > 0) {
        state=AbstractFileState::play;
        walltime = getCurrentMsTimestamp();
        reftime = walltime - target_mstimestamp_;
    
        valkkafslogger.log(LogLevel::debug) << "FileCacheThread : playStreams: reftime: " << reftime << std::endl;
        if (next) {
            valkkafslogger.log(LogLevel::debug) << "FileCacheThread : playStreams: next: " << *next << std::endl;
        }
        
        if (send_state) {
            state_setupframe.mstimestamp = walltime;
            state_setupframe.sub_type = SetupFrameType::stream_state;
            state_setupframe.stream_state = state;
            sendSetupFrames(&state_setupframe);
        }
    }
}


void FileCacheThread::setRefTimeAndStop(bool send_state) { // calculate reftime, send SetupFrame with stop parameter downstream
    walltime = getCurrentMsTimestamp();
    reftime = walltime - target_mstimestamp_;
    state = AbstractFileState::stop;
    
    if (send_state) {
        state_setupframe.mstimestamp = walltime;
        state_setupframe.sub_type = SetupFrameType::stream_state;
        state_setupframe.stream_state = state;
        sendSetupFrames(&state_setupframe);
    }
}


void FileCacheThread::seekStreams(long int mstimestamp_, bool clear, bool send_state) {
    int i;
    if (send_state) {
        state_setupframe.mstimestamp = walltime;
        state_setupframe.sub_type = SetupFrameType::stream_state;
        state_setupframe.stream_state = AbstractFileState::seek;
        sendSetupFrames(&state_setupframe);
        // this goes to downstream, all the way to OpenGLThread.  In OpenGLThread, the frame is cleared.
    }
    state = AbstractFileState::seek;
    
    // reftime = (t0 - t0_)
    target_mstimestamp_ = mstimestamp_;
    walltime = getCurrentMsTimestamp();
    // reftime = walltime - target_mstimestamp_; // reftime should be calculated only from the time instant when the frame is available and is ready to fire
    reftime = 0;
    next = NULL;
    
    valkkafslogger.log(LogLevel::debug) << "FileCacheThread : seekStreams : mintime : " << play_cache->getMinTime_() << std::endl;
    valkkafslogger.log(LogLevel::debug) << "FileCacheThread : seekStreams : maxtime : " << play_cache->getMaxTime_() << std::endl;
    
    // if the frame is available, go there immediately
    if ( ( (target_mstimestamp_ >= play_cache->getMinTime_()) and (target_mstimestamp_ <= play_cache->getMaxTime_()) ) and !clear) {
        valkkafslogger.log(LogLevel::debug) << "FileCacheThread : seekStreams : play_cache immediate seek" << std::endl;
        i=play_cache->keySeek(target_mstimestamp_); // could evoke a separate thread to do the job (in order to multiplex properly), but maybe its not necessary
        valkkafslogger.log(LogLevel::debug) << "FileCacheThread : seekStreams : play_cache seek done" << std::endl;
        if (i==0) { // Seek OK
            reftime = walltime - target_mstimestamp_;
            next = play_cache->pullNextFrame();
            if (!next) {
                valkkafslogger.log(LogLevel::fatal) << "FileCacheThread : seekStreams : WARNING : no next frame after seek" << std::endl;
            }
            // stopStreams(); // seeking a stream stops it
        }
    }
    else {
        reftime = 0; // reftime should be calculated only from the time instant when the frame is available and is ready to fire
    }
}
    

void FileCacheThread::clear() {
    reftime = 0;
    walltime = 0;
    next = NULL;
    state = AbstractFileState::stop;
    target_mstimestamp_ = 0;
}


void FileCacheThread::run() {
    std::unique_lock<std::mutex> lk(this->loop_mutex);
    bool ok;
    unsigned short subsession_index;
    long int dt;
    long int mstime, timeout, next_mstimestamp;
    long int save1mstime, save2mstime; // interrupt times
    int i;
    Frame* f;
    FrameFilter *ff;
    FileCacheSignalContext* signal_ctx;
    bool stream;
    
    mstime = getCurrentMsTimestamp();
    walltime = mstime;
    save1mstime = mstime;
    save2mstime = mstime;
    loop = true;
    next = NULL; // no next frame to be presented
    next_mstimestamp = 0;
    stream = false;
    
    while(loop) { // LOOP
        // always transmit all frames whose mstimestamp is less than target_mstimestamp_
        // if play mode, then update target_mstimestamp_
        // frame timestamp in wallclock time : t = t_ + reftime
        // calculate next Frame's wallclock time
        
        if (next and reftime > 0) { // there is a frame to await for
            next_mstimestamp = next->mstimestamp + reftime; // next Frame's timestamp in wallclock time
            if (state == AbstractFileState::stop) {
                timeout = Timeout::filecachethread;
            }
            else {
                timeout = std::min((long int)Timeout::filecachethread, (next_mstimestamp-walltime));
            }
        }
        else {
            timeout=Timeout::filecachethread;
            // timeout=300;
        }
        
        // std::cout << "FileCacheThread : run : timeout = " << timeout << std::endl;
        f=NULL;
        if (timeout > 0) { 
            f=infifo.read(timeout); // timeout == 0 blocks
        }
        // std::cout << "FileCacheThread : run : read ok " << std::endl;
        if (!f) { // TIMEOUT
            // std::cout << ": "<< this->name <<" timeout expired!" << std::endl;
        }
        else { // GOT FRAME // this must ALWAYS BE ACCOMPANIED WITH A RECYCLE CALL
            // Handle signal frames
            if (f->getFrameClass()==FrameClass::signal) {
                SignalFrame *signalframe = static_cast<SignalFrame*>(f);
                if (signalframe->signaltype==SignalType::filecachethread) {
                    FileCacheSignalContext signal_ctx = FileCacheSignalContext();
                    get_signal_context(signalframe, signal_ctx);
                    // signal_ctx = static_cast<FileCacheSignalContext*>(signalframe->custom_signal_ctx);
                    handleSignal(signal_ctx);
                    //delete signal_ctx; // custom signal context must be freed // not anymore
                }
            }
            else if (f->getFrameClass()==FrameClass::marker) {
                MarkerFrame *markerframe = static_cast<MarkerFrame*>(f);
                valkkafslogger.log(LogLevel::debug) << "FileCacheThread : got marker frame " << *markerframe << std::endl;
                if (markerframe->tm_start) {
                    valkkafslogger.log(LogLevel::debug) << "FileCacheThread : run : transmission start" << std::endl;
                    // tmp_cache->clear(); // most of frames in a block have already arrived to the FrameCache when we receive this frame (we're in a different thread)
                    // let the FrameCache do the clear instead
                }
                if (markerframe->tm_end) {
                    valkkafslogger.log(LogLevel::debug) << "FileCacheThread :  run : transmission end" << std::endl;
                    switchCache(); // start using the new FrameCache
                    next = NULL; // this refers to a cleared frame
                    // the client calls (1) the ValkkaFSReaderThread and (2) seekStreams of this thread
                    // When MarkerFrame with end flag arrives, the seek is activated
                    // We need: (a) target_mstimestamp_ (b) next Frame
                    if (target_mstimestamp_<=0) {
                        valkkafslogger.log(LogLevel::fatal) << "FileCacheThread : run : WARNING : got transmission end but seek time not set" << std::endl;
                    }
                    else {
                        valkkafslogger.log(LogLevel::debug) << "FileCacheThread :  run : play_cache seek with state " << int(state) << std::endl;
                        // so, the seek must be started from the previous i-frame, or not
                        // this depends on the state of the decoder.  If this is a "continuous-seek" (during play), then seeking from i-frame is not required
                        // could evoke seeks a separate thread to do the job (in order to multiplex properly), but maybe its not necessary
                        if (state == AbstractFileState::stop or state == AbstractFileState::seek) {
                            i=play_cache->keySeek(target_mstimestamp_); 
                        }
                        else if (state == AbstractFileState::play) {
                            i=play_cache->seek(target_mstimestamp_);
                        }
                        else {
                            valkkafslogger.log(LogLevel::fatal) << "FileCacheThread :  run : can't seek" << std::endl;
                            i=-1;
                        }
                        valkkafslogger.log(LogLevel::debug) << "FileCacheThread :  run : play_cache seek done" << std::endl;
                        if (i==0) { // Seek OK
                            valkkafslogger.log(LogLevel::debug) << "FileCacheThread :  run : play_cache seek OK" << std::endl;
                            next = play_cache->pullNextFrame(); // this will initiate the process of [pulling frames from the cache]
                            if (!next) {
                                valkkafslogger.log(LogLevel::fatal) << "FileCacheThread : run : WARNING : no next frame after seek" << std::endl;
                            }
                            if (reftime <= 0) { // seek is done.  match the current time to the requested frametime
                                walltime = getCurrentMsTimestamp();
                                reftime = walltime - target_mstimestamp_;
                            }
                        
                        } // Seek OK
                    }
                }
            }
            infifo.recycle(f); // always recycle
        } // GOT FRAME
        
        mstime = getCurrentMsTimestamp();
        
        if (state == AbstractFileState::play and reftime > 0) {
            walltime = mstime; // update wallclocktime (if play)
            target_mstimestamp_ = walltime - reftime;
        }
        
        /*
        if (next) {
            valkkafslogger.log(LogLevel::fatal) << "FileCacheThread : run : next - walltime " << next_mstimestamp - walltime << std::endl;
        }
        else {
            valkkafslogger.log(LogLevel::fatal) << "FileCacheThread : run : no next frame " << std::endl;
        }
        */
        
        // [pulling frames from the cache]
        while (next and ((next_mstimestamp - walltime) <= 0)) { // just send all frames at once
            // valkkafslogger.log(LogLevel::crazy) << "FileCacheThread :  run : transmit " << *next << std::endl;
            if (reftime <= 0) { // reftime has not been set, so set it at the first frame that gets sent downstream
                // std::cout << "FileCacheThread : setting reftime : " << mstime << ", " << target_mstimestamp_ << std::endl;
                // reftime = mstime - target_mstimestamp_;
                // stopStreams(); // with reftime set, this works and sends SetupFrame downstreams
                valkkafslogger.log(LogLevel::fatal) << "FileCacheThread : run : reftime not set" << std::endl;
            }
            
            i=safeGetSlot(next->n_slot, ff);
            if (i>=1) {
                if (!setup_frames[next->n_slot][next->subsession_index]) { // create a SetupFrame for decoder initialization
                    if (next->getFrameClass() == FrameClass::basic) { // check that the frame is BasicFrame
                        BasicFrame *basicf = static_cast<BasicFrame*>(next);
                        SetupFrame *setupf = new SetupFrame();
                        setupf->media_type = basicf->media_type;
                        setupf->codec_id = basicf->codec_id;
                        setupf->copyMetaFrom(basicf);
                        setup_frames[next->n_slot][next->subsession_index] = setupf;
                        #ifdef CACHESTREAM_VERBOSE
                        std::cout << "FileCacheThread : run : pushing SetupFrame " << *setupf << std::endl;
                        #endif
                        ff->run(setupf);
                    }
                }
                else {
                    // std::cout << "FileCacheThread : run : has setup frame " << *(setup_frames[next->subsession_index][next->n_slot]) << std::endl;
                }
                
                // modifying the frame here: it's just a pointer, so the frame in the underlying FrameCache will be modified as well
                // could create a copy here
                // valkkafslogger.log(LogLevel::crazy) << "FileCacheThread :  run : pushing frame " << *next << " distance to target=" << walltime - (next->mstimestamp + reftime) << std::endl;
                #ifdef CACHESTREAM_VERBOSE
                std::cout << "FileCacheThread :  run : pushing frame " << *next << " corrected timestamp = " << next->mstimestamp + reftime << std::endl;
                #endif
                
                // during the pushing, correct the frametime to walltime
                next->mstimestamp = next->mstimestamp + reftime;
                ff->run(next);
                next->mstimestamp = next->mstimestamp - reftime;
            }
            /*
            else {
                std::cout << "FileCacheThread: run: frame=" << *next << std::endl; // marker frames appear here..
            }
            */
            next=play_cache->pullNextFrame(); // new next frame
            if (next) {
                // valkkafslogger.log(LogLevel::crazy) << "FileCacheThread :  run : new next frame " << *next << std::endl;
                next_mstimestamp = next->mstimestamp + reftime; // next Frame's timestamp in wallclock time
                //  (t + (t0-t0_)) - t0 = t + t0 -t0_ -t0 = t-t0
                // target_mstimestamp_ = next->mstimestamp; // target time in streamtime // NOPES
            }
            
            if (!next or ((next_mstimestamp-walltime)>0)) {
                // this loop is about to break: there is no next frame, or the target frame has been reached
                if (state == AbstractFileState::seek) {
                    #ifdef CACHESTREAM_VERBOSE
                    if (!next) {
                        std::cout << "FileCacheThread: no next frame" << std::endl;
                    }
                    if ((next_mstimestamp-walltime)>0) {
                        std::cout << "FileCacheThread: target reached.  next = " << next_mstimestamp << std::endl;
                    }
                    std::cout << "FileCacheThread: Switching from seek to stop" << std::endl;
                    #endif
                    stopStreams();
                }
            }
        }
        
        // old-style ("interrupt") signal handling
        if ((mstime-save1mstime) >= Timeout::filecachethread) { // time to check the signals..
            // std::cout << "FileCacheThread: run: interrupt, dt= " << mstime-save1mstime << std::endl;
            handleSignals();
            // std::cout << "FileCacheThread: loop =" << loop << std::endl;
            save1mstime=mstime;
        }
        if ((mstime-save2mstime) >= Timeout::filecachethread and loop) { // send message to the python side
            if (pyfunc) {
                // consider this:
                // requestStopCall() => sends signal for the main loop to exit
                // waitStopCall() => calls thread join
                // when that join is called, execution might be here and thread join conflicts with the python GIL
                // (loop is set to false _after_ reading the signal)
                // that's why this method is protected by loop_mutex 
                //
                //std::cout << "FileCacheThread : run : time emit : " << walltime << std::endl;
                PyGILState_STATE gstate;
                
                gstate = PyGILState_Ensure();
                
                PyObject* res;
                
                if (reftime > 0) {
                    res=PyObject_CallFunction(pyfunc, "l", walltime-reftime); // send current stream time
                }
                else {
                    res=PyObject_CallFunction(pyfunc, "l", 0); // no reference time set 
                }
                if (!res) {
                    valkkafslogger.log(LogLevel::fatal) << "FileCacheThread :  run: WARNING: python time callback failed" << std::endl;
                }
                PyGILState_Release(gstate);
                
                //std::cout << "FileCacheThread : run : time emit exit: " << walltime << std::endl;
                
            }
            save2mstime=mstime;
        }
        
    } // LOOP
}
    
void FileCacheThread::preRun() {
}
    
void FileCacheThread::postRun() {
    valkkafslogger.log(LogLevel::debug) << "FileCacheThread :  postRun" << std::endl;    
    if (pyfunc!=NULL) {
        Py_DECREF(pyfunc);
    }
    if (pyfunc2!=NULL) {
        Py_DECREF(pyfunc2);
    }
}



void FileCacheThread::sendSignal(FileCacheSignalContext signal_ctx) {
    std::unique_lock<std::mutex> lk(this->mutex);
    this->signal_fifo.push_back(signal_ctx);
}


void FileCacheThread::handleSignal(FileCacheSignalContext &signal_ctx) {
    switch (signal_ctx.signal) {
        case FileCacheSignal::exit:
            loop=false;
            break;
        case FileCacheSignal::register_stream:
            registerStream(signal_ctx.pars.file_stream_ctx);
            break;
        case FileCacheSignal::deregister_stream:
            deregisterStream(signal_ctx.pars.file_stream_ctx);
            break;
        case FileCacheSignal::stop_streams:
            stopStreams();
            break;
        case FileCacheSignal::play_streams:
            playStreams();
            break;
        case FileCacheSignal::clear:
            clear();
            break;    
        case FileCacheSignal::seek_streams:
            seekStreams(signal_ctx.pars.mstimestamp, signal_ctx.pars.clear);
            break;
    }
}


void FileCacheThread::handleSignals() {
    std::unique_lock<std::mutex> lk(this->mutex);
    // handle pending signals from the signals fifo
    for (auto it = signal_fifo.begin(); it != signal_fifo.end(); ++it) { // it == pointer to the actual object (struct SignalContext)
        handleSignal(*it);
    }
    signal_fifo.clear();
}


FrameFilter &FileCacheThread::getFrameFilter() {
    return fork;
}

void FileCacheThread::requestStopCall() {
    if (!this->has_thread) { return; } // thread never started
    if (stop_requested) { return; }    // can be requested only once
    stop_requested = true;

    // use the old-style "interrupt" way of sending signals
    FileCacheSignalContext signal_ctx;
    signal_ctx.signal = FileCacheSignal::exit;
    
    this->sendSignal(signal_ctx);
}


int FileCacheThread::safeGetSlot(SlotNumber slot, FrameFilter*& ff) { 
    // -1 = out of range, 0 = free, 1 = reserved // &* = modify pointer in-place
    FrameFilter* framefilter;
    valkkafslogger.log(LogLevel::crazy) << "FileCacheThread: safeGetSlot: " << slot << std::endl;
    try {
        framefilter=this->slots_[slot];
    }
    catch (std::out_of_range) {
        valkkafslogger.log(LogLevel::debug) << "FileCacheThread: safeGetSlot : slot " << slot << " is out of range! " << std::endl;
        ff=NULL;
        return -1;
    }
    if (!framefilter) {
        valkkafslogger.log(LogLevel::crazy) << "FileCacheThread: safeGetSlot : nothing at slot " << slot << std::endl;
        ff=NULL;
        return 0;
    }
    else {
        valkkafslogger.log(LogLevel::crazy) << "FileCacheThread: safeGetSlot : returning " << slot << std::endl;
        ff=framefilter;
        return 1;
    }
}


void FileCacheThread::registerStream(FileStreamContext &ctx) {
    FrameFilter* framefilter;
    switch (safeGetSlot(ctx.slot, framefilter)) {
        case -1: // out of range
            break;
        case 0: // slot is free
            this->slots_[ctx.slot] = ctx.framefilter;
            break;
        case 1: // slot is reserved
            break;
    }
}
    
    
void FileCacheThread::deregisterStream(FileStreamContext &ctx) {
    FrameFilter* framefilter;
    switch (safeGetSlot(ctx.slot, framefilter)) {
        case -1: // out of range
            break;
        case 0: // slot is free
            break;
        case 1: // slot is reserved
            this->slots_[ctx.slot] = NULL;
            break;
    }
}

void FileCacheThread::registerStreamCall(FileStreamContext &ctx) {
    SignalFrame f = SignalFrame();
    // FileCacheSignalContext* signal_ctx = new FileCacheSignalContext();
    FileCacheSignalContext signal_ctx = FileCacheSignalContext();

    // encapsulate parameters
    FileCacheSignalPars pars;
    pars.file_stream_ctx = ctx;
    
    // encapsulate signal and parameters into signal context of the frame
    signal_ctx.signal = FileCacheSignal::register_stream;
    signal_ctx.pars   = pars;
    
    // f.custom_signal_ctx = (void*)(signal_ctx);
    put_signal_context(&f, signal_ctx, SignalType::filecachethread);
    infilter.run(&f);
}


void FileCacheThread::deregisterStreamCall (FileStreamContext &ctx) {
    SignalFrame f = SignalFrame();
    // FileCacheSignalContext* signal_ctx = new FileCacheSignalContext();
    FileCacheSignalContext signal_ctx = FileCacheSignalContext();

    // encapsulate parameters
    FileCacheSignalPars pars;
    pars.file_stream_ctx = ctx;
    
    // encapsulate signal and parameters into signal context of the frame
    signal_ctx.signal = FileCacheSignal::deregister_stream;
    signal_ctx.pars   = pars;
    
    // f.custom_signal_ctx = (void*)(signal_ctx);
    put_signal_context(&f, signal_ctx, SignalType::filecachethread);
    infilter.run(&f);
}
    
    
void FileCacheThread::dumpCache() {
    tmp_cache->dump();
}

void FileCacheThread::stopStreamsCall() {
    SignalFrame f = SignalFrame();
    // FileCacheSignalContext* signal_ctx = new FileCacheSignalContext();
    FileCacheSignalContext signal_ctx = FileCacheSignalContext();

    // encapsulate parameters
    FileCacheSignalPars pars;
    
    // encapsulate signal and parameters into signal context of the frame
    signal_ctx.signal = FileCacheSignal::stop_streams;
    signal_ctx.pars   = pars;
    
    //f.custom_signal_ctx = (void*)(signal_ctx);
    put_signal_context(&f, signal_ctx, SignalType::filecachethread);
    infilter.run(&f);
}
    
void FileCacheThread::playStreamsCall() {
    SignalFrame f = SignalFrame();
    // FileCacheSignalContext* signal_ctx = new FileCacheSignalContext();
    FileCacheSignalContext signal_ctx = FileCacheSignalContext();

    // encapsulate parameters
    FileCacheSignalPars pars;
    
    // encapsulate signal and parameters into signal context of the frame
    signal_ctx.signal = FileCacheSignal::play_streams;
    signal_ctx.pars   = pars;
    
    // f.custom_signal_ctx = (void*)(signal_ctx);
    put_signal_context(&f, signal_ctx, SignalType::filecachethread);
    infilter.run(&f);
}


void FileCacheThread::clearCall() {
    SignalFrame f = SignalFrame();
    // FileCacheSignalContext* signal_ctx = new FileCacheSignalContext();
    FileCacheSignalContext signal_ctx = FileCacheSignalContext();

    // encapsulate parameters
    FileCacheSignalPars pars;
    
    // encapsulate signal and parameters into signal context of the frame
    signal_ctx.signal = FileCacheSignal::clear;
    signal_ctx.pars   = pars;
    
    // f.custom_signal_ctx = (void*)(signal_ctx);
    put_signal_context(&f, signal_ctx, SignalType::filecachethread);
    infilter.run(&f);
}


void FileCacheThread::seekStreamsCall(long int mstimestamp, bool clear) {
    SignalFrame f = SignalFrame();
    // FileCacheSignalContext* signal_ctx = new FileCacheSignalContext();
    FileCacheSignalContext signal_ctx = FileCacheSignalContext();
    
    // encapsulate parameters
    FileCacheSignalPars pars;
    pars.mstimestamp = mstimestamp;
    pars.clear       = clear;
    
    // encapsulate signal and parameters into signal context of the frame
    signal_ctx.signal = FileCacheSignal::seek_streams;
    signal_ctx.pars   = pars;
    
    // f.custom_signal_ctx = (void*)(signal_ctx);
    put_signal_context(&f, signal_ctx, SignalType::filecachethread);
    infilter.run(&f);
}



