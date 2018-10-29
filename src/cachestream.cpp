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
 *  @version 0.1
 *  
 *  @brief 
 */ 

#include "cachestream.h"


CacheFrameFilter::CacheFrameFilter(const char* name, FrameCache* framecache) : FrameFilter(name), framecache(framecache) {
};

void CacheFrameFilter::go(Frame* frame) {
  framecache->writeCopy(frame);
}




FrameCache::FrameCache(const char *name, FrameCacheContext ctx) : name(name), ctx(ctx), mintime_(999999999999), maxtime_(0), has_delta_frames(false) {
    state = cache.end();
}

FrameCache::~FrameCache() {
    clear();
}
 
bool FrameCache::writeCopy(Frame* f, bool wait) {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  
  mintime_=std::min(mintime_, f->mstimestamp);
  maxtime_=std::max(maxtime_, f->mstimestamp);
  has_delta_frames = (has_delta_frames or !(f->isSeekable())); // frames that describe changes to previous frames are present
  
  cache.push_back(f->getClone());  // push_back takes a copy of the pointer
    
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

void FrameCache::clear() {
    std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
    mintime_=999999999999;
    maxtime_=0;
    has_delta_frames=false;
    for (auto it=cache.begin(); it!=cache.end(); ++it) {
        delete *it;
    }
    state = cache.end();
}


void FrameCache::dump() {
    std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
    for (auto it=cache.begin(); it!=cache.end(); ++it) {
        // std::cout << "FrameCache : dump : " << name << " : " << **it << " [" << (*it)->dumpPayload() << std::endl;
        std::cout << "FrameCache : dump : " << name << " : " << **it;
        if ((*it)->isSeekable()) {
            std::cout << " * ";
        }
        std::cout << std::endl;
    }
}

bool FrameCache::isEmpty() {
    std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
    return (cache.size() < 1);
}


int FrameCache::seek(long int ms_streamtime_) { // return values: -1 = no frames at left, 1 = no frames at right, 0 = ok
    std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
    
    if (cache.size() < 1) {
        state = cache.end();
        return -1;
    }
    if (ms_streamtime_ < mintime_) {
        state = cache.end();
        return -1;
    }
    if (ms_streamtime_ > maxtime_) {
        state = cache.end();
        return 1;
    }
    
    state = cache.begin();
    
    if (has_delta_frames) { // key-frame based seek
        for(auto it=cache.begin(); it!=cache.end(); ++it) {
            if ((*it)->mstimestamp > ms_streamtime_) {
                break;
            }
            if ((*it)->isSeekable()) {
                state = it;
            }
        }
    }
    else {
        for(auto it=cache.begin(); it!=cache.end(); ++it) {
            if ((*it)->mstimestamp > ms_streamtime_) {
                break;
            }
            state = it;
        }
    }
    
    if (state == cache.end()) {
        return 1;
    }
    return 0;
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



CacheStream::CacheStream(const char *name, FrameCacheContext cache_ctx) : cache(name, cache_ctx), infilter(name, &cache), target_mstimestamp_(-1), frame_mstimestamp_(-1), reftime(-1), state(AbstractFileState::none) {
}

CacheStream::~CacheStream() {    
}


void CacheStream::seek(long int ms_streamtime_) {
    setRefMstime(ms_streamtime_);
  
#ifdef FILE_VERBOSE  
  std::cout << "CacheStream : seek : seeking to " << ms_streamtime_ << std::endl;
#endif
    int i=cache.seek(ms_streamtime_);
    filethreadlogger.log(LogLevel::debug) << "CacheStream : seek returned " << std::endl;
    if (i!=0) {
        state=AbstractFileState::stop;
        return;
    }
    
    state=AbstractFileState::seek;
    target_mstimestamp_=ms_streamtime_;
    
    long int mstimeout = pullNextFrame(); // sends next_frame, reads new next_frame from the cache, returns timestamp of next frame
    
    if (mstimeout < 0) {
        state=AbstractFileState::stop;
        return;
    }
    
    // TODO: for receiving frame blocks, check here for the block marker frames and act accordingly
}


long int CacheStream::pullNextFrame() { 
    /*
    TODO: 
    - transmit current frame, pull next frame
    - recurse, if timeout == 0
    */
    std::cout << "CacheStream : pullNextFrame : transmitting " << *next_frame << std::endl;
    next_frame = cache.pullNextFrame();
    
    if (!next_frame) {
        return -1;
    }
    
    return std::max((long int)0, target_mstimestamp_- next_frame->mstimestamp);
}


long int CacheStream::update(long int mstimestamp) {
    long int timeout;
    
    if (state==AbstractFileState::stop) {
        return -1;
    }
    if (state==AbstractFileState::play or state==AbstractFileState::seek) { // when playing, target time changes ..
        target_mstimestamp_=mstimestamp-reftime;
    }
    timeout=pullNextFrame(); // for play and seek
    return timeout;
}





FileCacheThread::FileCacheThread(const char *name) : AbstractFileThread(name), stream(name) {
}


FileCacheThread::~FileCacheThread() {
}
        
void FileCacheThread::run() { // TODO
}
    
void FileCacheThread::preRun() { // TODO
}
    
void FileCacheThread::postRun() { // TODO
}
    
void FileCacheThread::sendSignal(FileCacheSignalContext signal_ctx) { // TODO
}
    
void FileCacheThread::handleSignals() { // TODO
}

void FileCacheThread::registerStream(FileCacheSignalContext &ctx) { // TODO
}
    
    
void FileCacheThread::deregisterStream(FileCacheSignalContext &ctx) { // TODO
}
  
void FileCacheThread::registerStreamCall(FileCacheSignalContext &ctx) { // TODO
}
    
void FileCacheThread::deregisterStreamCall (FileCacheSignalContext &ctx) { // TODO
}
    
const CacheFrameFilter& FileCacheThread::getFrameFilter() {
    return stream.getFrameFilter();
}
    
void FileCacheThread::requestStopCall() { // TODO
    
}


