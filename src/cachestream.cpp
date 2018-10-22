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




FrameCache::FrameCache(const char *name, FrameCacheContext ctx) : name(name), ctx(ctx) {
    //TODO: init cache, mutex, condition, ready_condition
}

FrameCache::~FrameCache() {
}
    
bool FrameCache::writeCopy(Frame* f, bool wait) { // TODO
    return False;
}
 
/*
Frame* FrameCache::read(unsigned short int mstimeout) { // TODO: do we need this?
    return NULL;
}
*/

void FrameCache::dumpFifo() { // TODO
}

bool FrameCache::isEmpty() { // TODO
}



CacheStream::CacheStream(const char *name, FrameCacheContext cache_ctx) : mintime_(0), maxtime_(0), cache(name, cache_ctx), infilter(name, &cache) {
}

CacheStream::~CacheStream() {    
}


void CacheStream::seek(long int ms_streamtime_) { // TODO
}


long int CacheStream::pullNextFrame() {
    return 0;
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


