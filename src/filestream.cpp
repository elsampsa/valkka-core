/*
 * filestream.cpp :
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
 *  @file    filestream.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.0.2 
 *  
 *  @brief 
 */ 

#include "filestream.h"



AbstractFileStream::AbstractFileStream() : reftime(0), target_mstimestamp_(0), frame_mstimestamp_(0), state(AbstractFileState::none), next_frame(NULL) {
}


AbstractFileStream::~AbstractFileStream() {
}


void AbstractFileStream::setRefMstime(long int ms_streamtime_) {
    reftime=(getCurrentMsTimestamp() - ms_streamtime_); 
}


void AbstractFileStream::play() {
    setRefMstime(target_mstimestamp_);
    state=AbstractFileState::play;
}


void AbstractFileStream::stop() {
    state=AbstractFileState::stop;
}


long int AbstractFileStream::update(long int mstimestamp) {
    long int timeout;
    if (state==AbstractFileState::stop) {
        return Timeout::filethread;
    }
    if (state==AbstractFileState::play or state==AbstractFileState::seek) { // when playing, target time changes ..
        target_mstimestamp_=mstimestamp-reftime;
    }
    timeout=pullNextFrame(); // for play and seek

    /*
    if (state==AbstractFileState::seek and (frame_mstimestamp_>=target_mstimestamp_)) {
    state=AbstractFileState::stop;
    timeout=Timeout::filethread;
    }
    */

    return timeout;
}


void AbstractFileStream::seek(long int ms_streamtime_) {
    setRefMstime(ms_streamtime_);
  
#ifdef FILE_VERBOSE  
  std::cout << "AbstractFileStream : seek : seeking to " << ms_streamtime_ << std::endl;
#endif
  
    int i = 0;
    // TODO: do the seek : set next_frame to correct positions
  
    filethreadlogger.log(LogLevel::debug) << "AbstractFileStream : seek returned " << std::endl;
    if (i<0) {
        state=AbstractFileState::stop;
        // TODO: send an info frame indicating end
        return;
    }
    
    if (i<0) {
        state=AbstractFileState::stop;
        // TODO: send an info frame indicating stream end
        return;
    }
    
  state=AbstractFileState::seek;
  target_mstimestamp_=ms_streamtime_;
  // frame_mstimestamp_ =(long int)avpkt->pts;
  // TODO: set frame_mstimestamp_
}



long int AbstractFileStream::pullNextFrame() {
    // presents the current frame ("old next_frame"), whose timestamp is frame_mstimestamp_
    // loads / sets the next_frame
    // return timestamp of next frame in wallclock time
    long int dt;
    int i;
    #ifdef FILE_VERBOSE  
    std::cout << "AbstractFileStream: pullNextFrame:                     " <<  std::endl;
    std::cout << "AbstractFileStream: pullNextFrame: reftime            : " << reftime << std::endl;
    std::cout << "AbstractFileStream: pullNextFrame: target_mstimestamp_: " << target_mstimestamp_ << std::endl;   
    std::cout << "AbstractFileStream: pullNextFrame: frame_mstimestamp_ : " << frame_mstimestamp_ << std::endl;          // frame, stream time
    std::cout << "AbstractFileStream: pullNextFrame:                     " << std::endl;
    #endif
    
    dt=frame_mstimestamp_-target_mstimestamp_;
    if ( dt>0 ) { // so, this has been called in vain.. must wait still
    #ifdef FILE_VERBOSE  
        std::cout << "AbstractFileStream: pullNextFrame: return timeout " << dt << std::endl;
    #endif
        return frame_mstimestamp_+reftime;
    }
    else if ( !next_frame ) { // no next_frame set
    #ifdef FILE_VERBOSE  
        std::cout << "AbstractFileStream: pullNextFrame: no next_frame" << std::endl;
    #endif
        // TODO: return .. what?
        return 0;
    }
    else {
        // use out_frame auxiliary frame for writing in the output framefilter
        // copy meta information
        // payload should be just a pointer to the frames "owned" by AbstractFileStream
        // .. as the framefifo performs copy-on-write
        
        out_frame.reset();
        /* // set the metainfo
        out_frame.n_slot    =ctx.slot;
        out_frame.codec_id  =codec_id;
        out_frame.media_type=media_type;
        */
        
        out_frame.mstimestamp=frame_mstimestamp_+reftime; // stream time.. let's fix that to wallclock time
        
        #ifdef FILE_VERBOSE  
        std::cout << "AbstractFileStream: pullNextFrame: sending frame: " << out_frame << std::endl;
        #endif
        // push the frame forward.. somehow
        // ctx.framefilter->run(&out_frame);
        
        // TODO: get the next frame
        // set frame_mstimestamp_, next_frame
        // if frame_mstimestamp_ is less than same, recurse this ..?
        
        /*
        if (i<0) {
            state=AbstractFileState::stop; // TODO: send an infoframe indicating that stream has finished
            return Timeout::filethread;
        }
        if (((long int)avpkt->pts)<=frame_mstimestamp_) { // same timestamp!  recurse (typical for sequences: sps, pps, keyframe)
            #ifdef FILE_VERBOSE  
            std::cout << "AbstractFileStream: pullNextFrame: recurse: " << std::endl;
            #endif
            dt=pullNextFrame();
        }
        else {
            if (state==AbstractFileState::seek) {
                state=AbstractFileState::stop;
            }
            dt=frame_mstimestamp_-target_mstimestamp_;
            return 
        }
        */
        
        return 0;
    }    
}


AbstractFileThread::AbstractFileThread(const char *name, FrameFifoContext fifo_ctx) : Thread(name), infifo(name,fifo_ctx), infilter(name,&infifo) {
}

AbstractFileThread::~AbstractFileThread() { // TODO : a model method
}


/*
 * while (loop):
 * 
 *      if ( (next_pts - mstime) <= 0)
 * 
 *          next_pts = AbstractFileStream->update(mstime)
 *              --> if called too early, returns
 *              => next_pts = AbstractFileStream->pullNextFrame()
 *                  => writes frame into FrameFilter
 *                  => take next time frame .. if frames with same timestamp, call recursively
 *                  => return next_pts
 *                  => set internal state : self.next_frame
 *          --> next_pts in wallclock time
 *          --> when seeking, next_pts is smaller than mstime for many consecutive frames  
 * 
 *      mstime = getCurrentMsTimestamp()
 *      --> handle signals if enough time has passed
 * 
 *      (refresh mstime)
 * 
 *      timeout = next_pts - mstime
 *      --> read incoming FrameFifo with timeout
 *      --> if frame, call AbstractFileStream->pushFrame(frame)
 *          => AbstractFileStream caches the frame
 * 
 *      (refresh mstime)
 */
void AbstractFileThread::run() { // TODO : a model method
}

void AbstractFileThread::preRun() { // TODO : a model method
}

void AbstractFileThread::postRun() { // TODO : a model method
}
