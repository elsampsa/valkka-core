/*
 * cachestream.cpp :
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
 *  @file    cachestream.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.10.0 
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
        std::cout << "FrameCache: writeCopy: got marker frame " << *markerframe << std::endl;
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
        std::cout << "FrameCache: "<<name<<" writeCopy : timing : inserting frame " << dt << " ms late" << std::endl;
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
    frame_cache_1("cache_1"), frame_cache_2("cache_2"), // define the filterchain from end to beginning
    cache_filter_1(name, &frame_cache_1), cache_filter_2(name, &frame_cache_2),
    switchfilter(name, &cache_filter_1, &cache_filter_2), // by default, frames are going into frame_cache_1
    typefilter(name, FrameClass::marker, &infilter), // infilter comes from the mother class
    fork(name, &switchfilter, &typefilter),
    play_cache(&frame_cache_2),
    tmp_cache(&frame_cache_1),
    callback(NULL), target_mstimestamp_(0), pyfunc(NULL), pyfunc2(NULL), next(NULL), reftime(0), walltime(0), state(AbstractFileState::stop)
    {
    /*
    Reservoir &res = infifo.getReservoir(FrameClass::signal); // TODO: move this into macro
    for(auto it=res.begin(); it!=res.end(); ++it) {
        SignalFrame* f = static_cast<SignalFrame*>(*it);
        f->custom_signal_ctx = (void*)(new FileCacheSignalContext()); 
    }
    */
    // init_signal_frames(infifo, FileCacheSignalContext); // nopes .. doesnt' make any sense .. at FrameFifo::writeCopy
    this->slots_.resize(I_MAX_SLOTS+1,NULL);
    this->setup_frames.resize(I_MAX_SLOTS+1);
    for (auto it_slot = setup_frames.begin(); it_slot != setup_frames.end(); ++it_slot) {
        it_slot->resize(I_MAX_SUBSESSIONS+1, NULL);
    }
    // SetupFrame *f = setup_frames[255][4]; // overflow
    // SetupFrame *f = setup_frames[4][255]; // correct order of indexes is this
}


FileCacheThread::~FileCacheThread() {
    /*
    Reservoir &res = infifo.getReservoir(FrameClass::signal); // TODO: move this into macro
    for(auto it=res.begin(); it!=res.end(); ++it) {
        SignalFrame* f = static_cast<SignalFrame*>(*it);
        delete (FileCacheSignalContext*)(f->custom_signal_ctx); // delete object of the correct type
    }
    */
    // clear_signal_frames(infifo, FileCacheSignalContext);
    
    for (auto it_slot=setup_frames.begin(); it_slot!=setup_frames.end(); ++it_slot) {
        for (auto it_subs=it_slot->begin(); it_subs!=it_slot->end(); ++it_subs) {
            if (*it_subs) {
                delete *it_subs;
            }
        }
    } // yes, I know, this sucks
    
    if (pyfunc!=NULL) {
        Py_DECREF(pyfunc);
    }
    if (pyfunc2!=NULL) {
        Py_DECREF(pyfunc2);
    }
}

void FileCacheThread::setCallback(void func(long int)) {
    callback = func;
}

void FileCacheThread::setPyCallback(PyObject* pobj) {
    // pass here, say "signal.emit" or a function/method that accepts single argument
    if (PyCallable_Check(pobj)) { // https://docs.python.org/3/c-api/type.html#c.PyTypeObject
        Py_INCREF(pobj);
        pyfunc=pobj;
    }
    else {
        std::cout << "TestThread: setPyCallback: needs python callable for emitting current time" << std::endl;
        pyfunc=NULL;
    }
}

void FileCacheThread::setPyCallback2(PyObject* pobj) {
    // pass here, say "signal.emit" or a function/method that accepts single argument
    if (PyCallable_Check(pobj)) { // https://docs.python.org/3/c-api/type.html#c.PyTypeObject
        Py_INCREF(pobj);
        pyfunc2=pobj;
    }
    else {
        std::cout << "TestThread: setPyCallback2: needs python callable for emitting loaded frame time limits" << std::endl;
        pyfunc2=NULL;
    }
}



void FileCacheThread::switchCache() {
    if (play_cache==&frame_cache_1) {
        play_cache=&frame_cache_2; // this thread is manipulating frame_cache_2
        tmp_cache=&frame_cache_1;
        switchfilter.set1(); // new frames are cached to frame_cache_1
        frame_cache_1.clear(); // .. but before that, clear it
    }
    else { // vice versa
        play_cache=&frame_cache_1;
        tmp_cache=&frame_cache_2;
        switchfilter.set2();
        frame_cache_2.clear();
    }
    
    if (pyfunc2) {
        std::cout << "FileCacheThread : switchCache : evoke python callback : " << std::endl;
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
            std::cout << "FileCacheThread: switchCache: WARNING: python callback failed" << std::endl;
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
    std::cout << "FileCacheThread : sendSetupFrames : " << std::endl;
    for (auto it=slots_.begin(); it!=slots_.end(); ++it) {
        if (*it) {
            std::cout << "FileCacheThread : sendSetupFrames : " << cc << " : " << *f << std::endl;
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
        
        if (send_state) {
            state_setupframe.mstimestamp = walltime;
            state_setupframe.sub_type = SetupFrameType::stream_state;
            state_setupframe.stream_state = state;
            sendSetupFrames(&state_setupframe);
        }
    }
}

void FileCacheThread::seekStreams(long int mstimestamp_, bool send_state) {
    int i;
    if (send_state) {
        state_setupframe.mstimestamp = walltime;
        state_setupframe.sub_type = SetupFrameType::stream_state;
        state_setupframe.stream_state = AbstractFileState::seek;
        sendSetupFrames(&state_setupframe);
    }
    stopStreams(); // seeking a stream stops it
    // reftime = (t0 - t0_)
    target_mstimestamp_ = mstimestamp_;
    walltime = getCurrentMsTimestamp();
    reftime = walltime - target_mstimestamp_;
    next=NULL;
    
    std::cout << "FileCacheThread : seekStreams : mintime : " << play_cache->getMinTime_() << std::endl;
    std::cout << "FileCacheThread : seekStreams : maxtime : " << play_cache->getMaxTime_() << std::endl;
    
    // this if condition is unnecessary : the python method checks first the blocktable and only after that uses seekStreamsCall
    // .. let's keep it here just for cpp debugging purposes
    if ( (target_mstimestamp_ >= play_cache->getMinTime_()) and (target_mstimestamp_ <= play_cache->getMaxTime_()) ) {
        std::cout << "FileCacheThread : seekStreams : play_cache seek" << std::endl;
        i=play_cache->keySeek(target_mstimestamp_); // could evoke a separate thread to do the job (in order to multiplex properly), but maybe its not necessary
        std::cout << "FileCacheThread : seekStreams : play_cache seek done" << std::endl;
        if (i==0) { // Seek OK
            next=play_cache->pullNextFrame();
            if (!next) {
                std::cout << "FileCacheThread : seekStreams : WARNING : no next frame after seek" << std::endl;
            }
            // walltime=reftime+target_mstimestamp_; // a bit awkward .. recover the walltime when seekStreams was called
        }
    }
    
    // i=play_cache->seek(target_mstimestamp_); // 0 = ok    
    /*
    std::cout << "FileCacheThread : seekStreams: play_cache->seek returned " << i << std::endl;
    if (i!=0) {
        if (callback) {
            (*callback)(mstimestamp);
        }
        if (pyfunc!=NULL) {
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();
            PyObject_CallFunction(pyfunc, "l", mstimestamp);    
            PyGILState_Release(gstate);
        }
        return;
    }
    */
    
    // next = play_cache->pullNextFrame();
    // std::cout << "FileCacheThread : seekStreams: got frame " << *next << std::endl;
    // return;
    
    /*
    while(next and (next->mstimestamp <= target_mstimestamp_)) {
        i=safeGetSlot(next->n_slot, ff);
        if (i>=1) {
            ff->run(next);
        }
        next=play_cache->pullNextFrame();
    }
    */
}
    



        
void FileCacheThread::run() {
    // TODO: send SetupFrame as the first frame
    // TODO: define on render context, that last frame should be kept
    // pulls next frame from the stream (CacheStream) and sends it to the correct FrameFilter
    // handle the sending of SetupFrame(s) (that are used to initialize decoders)
    // For each received Frame, write them to FrameFilter at slots_[n_slot]
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
    loop=true;
    next=NULL; // no next frame to be presented
    next_mstimestamp=0;
    stream=false;
    
    while(loop) { // LOOP
        // always transmit all frames whose mstimestamp is less than target_mstimestamp_
        // if play mode, then update target_mstimestamp_
        // frame timestamp in wallclock time : t = t_ + reftime
        // calculate next Frame's wallclock time
        
        if (next) { // there is a frame to await for
            // TODO: what to do when seek ended?
            // std::cout << "FileCacheThread : run : has next frame " << *next << std::endl;
            // target_mstimestamp_ = next->mstimestamp; // target time in streamtime // NO NO NO
            next_mstimestamp = next->mstimestamp + reftime; // next Frame's timestamp in wallclock time
            // timeout=std::max((long int)0,(next_mstimestamp-walltime)); // timeout: timestamp - wallclock time.  For seek, wallclock time is frozen
            timeout=std::min((long int)300,(next_mstimestamp-walltime));
        }
        else {
            // timeout=Timeout::filecachethread;
            timeout=300;
        }
        
        // std::cout << "FileCacheThread : run : timeout = " << timeout << std::endl;
        f=NULL;
        if (timeout>0) { 
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
                signal_ctx = static_cast<FileCacheSignalContext*>(signalframe->custom_signal_ctx);
                handleSignal(*signal_ctx);
                delete signal_ctx; // custom signal context must be freed
            }
            else if (f->getFrameClass()==FrameClass::marker) {
                MarkerFrame *markerframe = static_cast<MarkerFrame*>(f);
                std::cout << "got marker frame " << *markerframe << std::endl;
                if (markerframe->tm_start) {
                    std::cout << "FileCacheThread : run : transmission start" << std::endl;
                    // tmp_cache->clear(); // most of frames in a block have already arrived to the FrameCache when we receive this frame (we're in a different thread)
                    // let the FrameCache do the clear instead
                }
                if (markerframe->tm_end) {
                    std::cout << "FileCacheThread : run : transmission end" << std::endl;
                    switchCache(); // start using the new FrameCache
                    // the client calls (1) the ValkkaFSReaderThread and (2) seekStreams of this thread
                    // When MarkerFrame with end flag arrives, the seek is activated
                    // We need: (a) target_mstimestamp_ (b) next Frame
                    if (target_mstimestamp_<=0) {
                        std::cout << "FileCacheThread : run : WARNING : got transmission end but seek time not set" << std::endl;
                    }
                    else {
                        std::cout << "FileCacheThread : run : play_cache seek" << std::endl;
                        // so, the seek must be started from the previous i-frame, or not
                        // this depends on the state of the decoder.  If this is a "continuous-seek" (during play), then seeking from i-frame is not required
                        // could evoke seeks a separate thread to do the job (in order to multiplex properly), but maybe its not necessary
                        if (state == AbstractFileState::stop) {
                            i=play_cache->keySeek(target_mstimestamp_); 
                        }
                        else if (state == AbstractFileState::play) {
                            i=play_cache->seek(target_mstimestamp_);
                        }
                        else {
                            std::cout << "FileCacheThread : run : can't seek" << std::endl;
                            i=-1;
                        }
                        std::cout << "FileCacheThread : run : play_cache seek done" << std::endl;
                        if (i==0) { // Seek OK
                            next=play_cache->pullNextFrame();
                            if (!next) {
                                std::cout << "FileCacheThread : run : WARNING : no next frame after seek" << std::endl;
                            }
                            // walltime=reftime+target_mstimestamp_; // a bit awkward .. recover the walltime when seekStreams was called
                        } // Seek OK
                    }
                }
            }
            infifo.recycle(f); // always recycle
        } // GOT FRAME
        
        mstime = getCurrentMsTimestamp();
        
        if (state==AbstractFileState::play) {
            walltime = mstime; // update wallclocktime (if play)
            target_mstimestamp_ = walltime - reftime;
        }
        
        /*
        if ((next_mstimestamp-walltime)<=0) { // frames that are late or at the current wallclock time
            // TODO: transmit next
            next=play_cache->pullNextFrame(); // new next frame
        }
        */
        while (next and ((next_mstimestamp-walltime)<=0)) { // just send all frames at once
            std::cout << "FileCacheThread : run : transmit " << *next << std::endl;
            i=safeGetSlot(next->n_slot, ff);
            if (i>=1) {
                if (!setup_frames[next->subsession_index][next->n_slot]) { // create a SetupFrame for decoder initialization
                    if (next->getFrameClass() == FrameClass::basic) { // check that the frame is BasicFrame
                        BasicFrame *basicf = static_cast<BasicFrame*>(next);
                        SetupFrame *setupf = new SetupFrame();
                        setupf->media_type = basicf->media_type;
                        setupf->codec_id = basicf->codec_id;
                        setupf->copyMetaFrom(basicf);
                        setup_frames[next->subsession_index][next->n_slot] = setupf;
                        std::cout << "FileCacheThread : run : pushing frame " << *setupf << std::endl;
                        ff->run(setupf);
                    }
                }
                // modifying the frame here: it's just a pointer, so the frame in the underlying FrameCache will be modified as well
                // could create a copy here
                std::cout << "FileCacheThread : run : pushing frame " << *next << " distance to target=" << walltime - (next->mstimestamp + reftime) << std::endl;
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
                std::cout << "FileCacheThread : run : new next frame " << *next << std::endl;
                next_mstimestamp = next->mstimestamp + reftime; // next Frame's timestamp in wallclock time
                //  (t + (t0-t0_)) - t0 = t + t0 -t0_ -t0 = t-t0
                // target_mstimestamp_ = next->mstimestamp; // target time in streamtime // NOPES
            }
        }
        
        // old-style ("interrupt") signal handling
        if ((mstime-save1mstime)>=Timeout::filecachethread) { // time to check the signals..
            // std::cout << "FileCacheThread: run: interrupt, dt= " << mstime-save1mstime << std::endl;
            handleSignals();
            save1mstime=mstime;
        }
        if ((mstime-save2mstime)>=300) { // send message to the python side // TODO: one should be able to adjust this frequency .. 1000 ms is too big
            //std::cout << "FileCacheThread : run : python callback interrupt" << std::endl;
            if (pyfunc) {
                // std::cout << "FileCacheThread : run : evoke python callback : " << walltime << std::endl;
                PyGILState_STATE gstate;
                gstate = PyGILState_Ensure();
                PyObject* res;
                
                if (reftime>0) {
                    res=PyObject_CallFunction(pyfunc, "l", walltime-reftime); // send current stream time
                }
                else {
                    res=PyObject_CallFunction(pyfunc, "l", 0); // no reference time set 
                }
                if (!res) {
                    std::cout << "FileCacheThread: run: WARNING: python callback failed" << std::endl;
                }
                PyGILState_Release(gstate);
            }
            save2mstime=mstime;
        }
        
    } // LOOP
}
    
void FileCacheThread::preRun() {
}
    
void FileCacheThread::postRun() {
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
        case FileCacheSignal::seek_streams:
            seekStreams(signal_ctx.pars.mstimestamp);
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
    FileCacheSignalContext* signal_ctx = new FileCacheSignalContext();
    
    // encapsulate parameters
    FileCacheSignalPars pars;
    pars.file_stream_ctx = ctx;
    
    // encapsulate signal and parameters into signal context of the frame
    signal_ctx->signal = FileCacheSignal::register_stream;
    signal_ctx->pars   = pars;
    
    f.custom_signal_ctx = (void*)(signal_ctx);
    infilter.run(&f);
}


void FileCacheThread::deregisterStreamCall (FileStreamContext &ctx) {
    SignalFrame f = SignalFrame();
    FileCacheSignalContext* signal_ctx = new FileCacheSignalContext();
    
    // encapsulate parameters
    FileCacheSignalPars pars;
    pars.file_stream_ctx = ctx;
    
    // encapsulate signal and parameters into signal context of the frame
    signal_ctx->signal = FileCacheSignal::deregister_stream;
    signal_ctx->pars   = pars;
    
    f.custom_signal_ctx = (void*)(signal_ctx);
    infilter.run(&f);
}
    
    
void FileCacheThread::dumpCache() {
    tmp_cache->dump();
}

void FileCacheThread::stopStreamsCall() {
    SignalFrame f = SignalFrame();
    FileCacheSignalContext* signal_ctx = new FileCacheSignalContext();
    
    // encapsulate parameters
    FileCacheSignalPars pars;
    
    // encapsulate signal and parameters into signal context of the frame
    signal_ctx->signal = FileCacheSignal::stop_streams;
    signal_ctx->pars   = pars;
    
    f.custom_signal_ctx = (void*)(signal_ctx);
    infilter.run(&f);
}
    
void FileCacheThread::playStreamsCall() {
    SignalFrame f = SignalFrame();
    FileCacheSignalContext* signal_ctx = new FileCacheSignalContext();
    
    // encapsulate parameters
    FileCacheSignalPars pars;
    
    // encapsulate signal and parameters into signal context of the frame
    signal_ctx->signal = FileCacheSignal::play_streams;
    signal_ctx->pars   = pars;
    
    f.custom_signal_ctx = (void*)(signal_ctx);
    infilter.run(&f);
}

void FileCacheThread::seekStreamsCall(long int mstimestamp) {
    SignalFrame f = SignalFrame();
    FileCacheSignalContext* signal_ctx = new FileCacheSignalContext();
    
    // encapsulate parameters
    FileCacheSignalPars pars;
    pars.mstimestamp = mstimestamp;
    
    // encapsulate signal and parameters into signal context of the frame
    signal_ctx->signal = FileCacheSignal::seek_streams;
    signal_ctx->pars   = pars;
    
    f.custom_signal_ctx = (void*)(signal_ctx);
    infilter.run(&f);
}

    

