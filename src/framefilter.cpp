/*
 * framefilter.cpp :
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
 *  @file    framefilter.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.4.6 
 *  
 *  @brief 
 */ 

#include "framefilter.h"
#include "tools.h"

// #define TIMESTAMPFILTER_DEBUG // keep this commented

FrameFilter::FrameFilter(const char* name, FrameFilter* next) : name(name), next(next) {
};

FrameFilter::~FrameFilter() {
};
  
void FrameFilter::run(Frame* frame) {
  this->go(frame); // manipulate frame
  if (!this->next) { return; } // call next filter .. if there is any
  (this->next)->run(frame);
}
  

// subclass like this:
DummyFrameFilter::DummyFrameFilter(const char* name, bool verbose, FrameFilter* next) : FrameFilter(name,next), verbose(verbose) {
  // std::cout << ">>>>>>" << verbose << std::endl;
}
  
void DummyFrameFilter::go(Frame* frame) { 
  if (verbose) {
    // std::cout << "DummyFrameFilter : "<< this->name << " " << verbose << " : got frame : " << *(frame) << std::endl;
    std::cout << "DummyFrameFilter : "<< this->name << " : got frame : " << *(frame) << std::endl;
  }
}



InfoFrameFilter::InfoFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next) {
}

void InfoFrameFilter::go(Frame* frame) {
  std::cout << "InfoFrameFilter: " << name << " start dump>> " <<std::endl;
  std::cout << "InfoFrameFilter: FRAME   : "<< *(frame) << std::endl;
  std::cout << "InfoFrameFilter: PAYLOAD : [";
  std::cout << frame->dumpPayload();
  std::cout << "]" << std::endl;
  // std::cout << "InfoFrameFilter:<" << frame->dumpAVFrame() << ">" << std::endl;
  std::cout << "InfoFrameFilter: timediff: " << frame->mstimestamp-getCurrentMsTimestamp() << std::endl;
  std::cout << "InfoFrameFilter: " << name << " <<end dump   " <<std::endl;
  
}


BriefInfoFrameFilter::BriefInfoFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next) {
}

void BriefInfoFrameFilter::go(Frame* frame) {
  std::cout << "DummyFrameFilter : "<< this->name << " : got frame : " << *(frame) << " dT=" << frame->mstimestamp-getCurrentMsTimestamp() << std::endl;
}


ForkFrameFilter::ForkFrameFilter(const char* name, FrameFilter* next, FrameFilter* next2) : FrameFilter(name,next), next2(next2) {
}

void ForkFrameFilter::run(Frame* frame) {
  // std::cout << "ForkFrameFilter: run" << std::endl;
  this->go(frame); // manipulate frame
  if (!this->next) {
    }
  else {
    (this->next)->run(frame);
  }
  if (!this->next2) {
    }
  else {
    (this->next2)->run(frame);
  }
}


void ForkFrameFilter::go(Frame* frame) {
 // filterlogger.log(LogLevel::debug) << "ForkFrameFilter : "<< this->name << " : got frame : " << *(frame) << std::endl;
}



ForkFrameFilter3::ForkFrameFilter3(const char* name, FrameFilter* next, FrameFilter* next2, FrameFilter* next3) : FrameFilter(name,next), next2(next2), next3(next3) {
}

void ForkFrameFilter3::run(Frame* frame) {
  // std::cout << "ForkFrameFilter: run" << std::endl;
  this->go(frame); // manipulate frame
  if (!this->next) {
    }
  else {
    (this->next)->run(frame);
  }
  if (!this->next2) {
    }
  else {
    (this->next2)->run(frame);
  }
  if (!this->next3) {
    }
  else {
    (this->next3)->run(frame);
  }
}


void ForkFrameFilter3::go(Frame* frame) {
 // filterlogger.log(LogLevel::debug) << "ForkFrameFilter : "<< this->name << " : got frame : " << *(frame) << std::endl;
}



SlotFrameFilter::SlotFrameFilter(const char* name, SlotNumber n_slot, FrameFilter* next) : FrameFilter(name,next), n_slot(n_slot) {
}
    

void SlotFrameFilter::go(Frame* frame) {
  frame->n_slot=n_slot;
}



DumpFrameFilter::DumpFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next), count(0) {
}

void DumpFrameFilter::go(Frame* frame) { // du -h *.bin | grep "s0"
  std::string filename=std::string("packet_c")+
    std::to_string(count)+std::string("_s")+                      // packet count
    std::to_string(frame->subsession_index)+std::string("_")+    // subsession index
    std::to_string(frame->mstimestamp)+                          // timestamp 
    std::string(".bin");
  
  // std::cout << "DumpFrameFilter: writing "<< frame->payload.size() << " bytes" << std::endl;
  // std::cout << "DumpFrameFilter: payload : " << frame->dumpPayload() <<std::endl;
  
  std::ofstream fout(filename, std::ios::out | std::ofstream::binary);
  
  /*
  for(auto it=(frame->payload).begin(); it!=(frame->payload).end(); ++it) {
    std::cout << ">>" << (int)*it;
    fout << (char)*it;
  }
  */

  frame->dumpPayloadToFile(fout);
  // std::copy((frame->payload).begin(), (frame->payload).end(), std::ostreambuf_iterator<char>(fout));
  
  fout.close();
  count++;
}


CountFrameFilter::CountFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next), count(0) {
}

void CountFrameFilter::go(Frame* frame) { // du -h *.bin | grep "s0"
  count++;
  std::cout << "CountFrameFilter : got frame " << count << std::endl;
}


TimestampFrameFilter::TimestampFrameFilter(const char* name, FrameFilter* next, long int msdiff_max) : FrameFilter(name,next), msdiff_max(msdiff_max), mstime_delta(0) {
}


void TimestampFrameFilter::go(Frame* frame) {
  long int ctime, corrected, diff;
  
  ctime     =getCurrentMsTimestamp();           // current time
  corrected =frame->mstimestamp+mstime_delta;  // corrected timestamp
  diff      =corrected-ctime;                  // time difference between corrected and current time.  positive == frame in the future, mstime_delta must be set to negative

#ifdef TIMESTAMPFILTER_DEBUG
  std::cout << "TimestampFrameFilter: go: ctime, frame->mstimestamp, corrected, diff : " << ctime << " " << frame->mstimestamp << " " << corrected << " " << diff << std::endl;
#endif
  
  if ( std::abs(diff)>msdiff_max ) { // correct the correction..
    mstime_delta=mstime_delta-diff; // positive diff, mstime_delta must be subtracted
#ifdef TIMESTAMPFILTER_DEBUG
    std::cout << "TimestampFrameFilter: go: CHANGING mstime_delta to " << mstime_delta << std::endl;
#endif
  }
  
  frame->mstimestamp=(frame->mstimestamp+mstime_delta);
#ifdef TIMESTAMPFILTER_DEBUG
  std::cout << "TimestampFrameFilter: go: final frame->mstimestamp " << frame->mstimestamp << std::endl;
#endif
}


TimestampFrameFilter2::TimestampFrameFilter2(const char* name, FrameFilter* next, long int msdiff_max) : FrameFilter(name,next), msdiff_max(msdiff_max), mstime_delta(0), savedtimestamp(0) {
}


void TimestampFrameFilter2::go(Frame* frame) {
  long int ctime, corrected, diff;
  
  if ( (frame->mstimestamp-savedtimestamp)>DEFAULT_TIMESTAMP_RESET_TIME) { // reset the correction once in a minute
    mstime_delta=0;
    savedtimestamp=frame->mstimestamp;
#ifdef TIMESTAMPFILTER_DEBUG
    std::cout << "TimestampFrameFilter2: reset correction" << std::endl;
#endif
  }
  
  ctime     =getCurrentMsTimestamp();          // current time
  corrected =frame->mstimestamp+mstime_delta;  // corrected timestamp
  diff      =corrected-ctime;                  // time difference between corrected and current time.  positive == frame in the future, mstime_delta must be set to negative

#ifdef TIMESTAMPFILTER_DEBUG
  std::cout << "TimestampFrameFilter2: go: ctime, frame->mstimestamp, corrected, diff : " << ctime << " " << frame->mstimestamp << " " << corrected << " " << diff << std::endl;
#endif
  
  if ( std::abs(diff)>msdiff_max ) { // correct the correction..
    mstime_delta=mstime_delta-diff; // positive diff, mstime_delta must be subtracted
#ifdef TIMESTAMPFILTER_DEBUG
    std::cout << "TimestampFrameFilter2: go: CHANGING mstime_delta to " << mstime_delta << std::endl;
#endif
  }
  
  frame->mstimestamp=(frame->mstimestamp+mstime_delta);
#ifdef TIMESTAMPFILTER_DEBUG
  std::cout << "TimestampFrameFilter2: go: final frame->mstimestamp " << frame->mstimestamp << std::endl;
#endif
  
  
#ifdef PROFILE_TIMING
  long int dt=(ctime-frame->mstimestamp);
  // std::cout << "[PROFILE_TIMING] FrameSink: afterGettingFrame: sending frame at " << dt << " ms" << std::endl;
  if (dt>=50) {
    std::cout << "[PROFILE_TIMING] TimestampFrameFilter2: go : sending frame " << dt << " ms late" << std::endl;
  }
#endif
  
}



DummyTimestampFrameFilter::DummyTimestampFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next) {
}


void DummyTimestampFrameFilter::go(Frame* frame) {
  frame->mstimestamp=getCurrentMsTimestamp();
}



RepeatH264ParsFrameFilter::RepeatH264ParsFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next) {
}

void RepeatH264ParsFrameFilter::go(Frame* frame) {// TODO
}



GateFrameFilter::GateFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next), on(false), config_frames(true) {
}
  
void GateFrameFilter::go(Frame* frame) {
}
  
void GateFrameFilter::run(Frame* frame) {
  std::unique_lock<std::mutex> lk(mutex);
  this->go(frame); // manipulate frame
  if (!next) {return;}
  if (on) { // pass all frames if flag is set
    (this->next)->run(frame);
  }
  else if (frame->getFrameClass()==FrameClass::setup and config_frames) { // .. if flag is not set, pass still config frames if config_frames is set
    (this->next)->run(frame);  
  }
}
  
void GateFrameFilter::set() {
  std::unique_lock<std::mutex> lk(mutex);
  on=true;
}

void GateFrameFilter::unSet() {
  std::unique_lock<std::mutex> lk(mutex);
  on=false;
}

void GateFrameFilter::passConfigFrames() {
  std::unique_lock<std::mutex> lk(mutex);
  config_frames=true;
}

void GateFrameFilter::noConfigFrames() {
  std::unique_lock<std::mutex> lk(mutex);
  config_frames=false;
  
}



CachingGateFrameFilter::CachingGateFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next), on(false), setupframe(), got_setup(false) {
}
  
void CachingGateFrameFilter::go(Frame* frame) {
}

void CachingGateFrameFilter::run(Frame* frame) {
  std::unique_lock<std::mutex> lk(mutex);
  this->go(frame); // manipulate frame
  if (!next) {return;}
  if (on) { // passes all frames if flag is set
    (this->next)->run(frame);
  }
  if (frame->getFrameClass()==FrameClass::setup) {
    got_setup=true;
    setupframe = *(static_cast<SetupFrame*>(frame)); // create a cached copy of the SetupFrame
  }
}
  
void CachingGateFrameFilter::set() {
  std::unique_lock<std::mutex> lk(mutex);
  on=true;
  if (got_setup) {
    (this->next)->run(&setupframe); // emit the cached setupframe always when the gate is activated
  }
}

void CachingGateFrameFilter::unSet() {
  std::unique_lock<std::mutex> lk(mutex);
  on=false;
}



SetSlotFrameFilter::SetSlotFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next), n_slot(0) {
}

void SetSlotFrameFilter::go(Frame* frame) {
  std::unique_lock<std::mutex> lk(mutex);
  if (n_slot>0) {
    frame->n_slot=n_slot;
  }
}
  
void SetSlotFrameFilter::setSlot(SlotNumber n) {
  std::unique_lock<std::mutex> lk(mutex);
  n_slot=n;
}
  

TimeIntervalFrameFilter::TimeIntervalFrameFilter(const char* name, long int mstimedelta, FrameFilter* next) : FrameFilter(name,next), mstimedelta(mstimedelta) {
  prevmstimestamp=0;  
}

void TimeIntervalFrameFilter::go(Frame* frame) { // this does nothing
}


void TimeIntervalFrameFilter::run(Frame* frame) {
  // std::cout << std::endl << "TimeIntervalFrameFilter: " << std::endl;
  if (!this->next) { return; } // calls next filter .. if there is any
  // std::cout << std::endl << "TimeIntervalFrameFilter: mstimestamps=" << frame->mstimestamp << " " << prevmstimestamp << std::endl;
  // std::cout << std::endl << "TimeIntervalFrameFilter: delta       =" << (frame->mstimestamp-prevmstimestamp) << std::endl;
  if ( (frame->mstimestamp-prevmstimestamp)>=mstimedelta ) {
    // std::cout << std::endl << "TimeIntervalFrameFilter: WRITE" << std::endl;
    prevmstimestamp=frame->mstimestamp;
    (this->next)->run(frame);
  }
}



FifoFrameFilter::FifoFrameFilter(const char* name, FrameFifo* framefifo) : FrameFilter(name), framefifo(framefifo) {
};

void FifoFrameFilter::go(Frame* frame) {
  framefifo->writeCopy(frame);
}



BlockingFifoFrameFilter::BlockingFifoFrameFilter(const char* name, FrameFifo* framefifo) : FrameFilter(name), framefifo(framefifo) {
};

void BlockingFifoFrameFilter::go(Frame* frame) {
  // std::cout << "BlockingFifoFrameFilter: go" << std::endl;
  framefifo->writeCopy(frame,true);
  /*
  
  condition variable A: triggered when there are frames in the queue
  condition variable B: triggered when there are frames available in the stack
  
  this->mutex controls locking with "lk"
  
  server:
  framefifo.writeCopy     reserves mutex .. uses condition_variable.notify_one(lk) 
  
  
  server with wait:
  
  framefifo.writeCopy
                          reserve mutex .. observe the state of the fifo .. if no more frames available in the stack, go to a waiting state .. wait for a condition variable B : condition_variable.wait(lk) releases the lock every now then
    
  client:
  framefifo.read          reserves mutex, trigger condition variable B.  Wait for condition variable A.
  
  
  waits for a condition variable.   condition_variable.wait(lk) releases the mutex for an instant .. imagine that it releases it every now and then ..
  
  
  ----
  
  server:
  framefifo.writeCopy    reserves mutex
  

  */
}


/** Interpolates from YUV to RGB with requested dimensions
 * 
 */
SwScaleFrameFilter::SwScaleFrameFilter(const char* name, int target_width, int target_height, FrameFilter* next) : FrameFilter(name,next), target_width(target_width), target_height(target_height), width(0), height(0), sws_ctx(NULL) {
  // shorthand
  AVFrame *output_frame =outputframe.av_frame;
  
  int nb =av_image_alloc(output_frame->data, output_frame->linesize, target_width, target_height, AV_PIX_FMT_RGB24, 1);
  if (nb<=0) {
    std::cout << "SwScaleFrameFilter: could not reserve frame " << std::endl;
    exit(5);
  }
  decoderlogger.log(LogLevel::debug) << "SwScaleFrameFilter: reserved " << nb << " bytes for the bitmap" << std::endl;
  
  // set correct outputframe.bmpars
  outputframe.av_frame->width =target_width;
  outputframe.av_frame->height=target_height;
  outputframe.av_pixel_format =AV_PIX_FMT_RGB24;
  outputframe.update(); // update outpuframe.bmpars using those values
}


SwScaleFrameFilter::~SwScaleFrameFilter() {
  if (!sws_ctx) {
  }
  else {
    sws_freeContext(sws_ctx);
  }
}


void SwScaleFrameFilter::run(Frame* frame) { // AVThread calls this ..
  // std::cout << "SwScaleFrameFilter: run" << std::endl;
  this->go(frame); // manipulate frame - in this case, scale from yuv to rgb
  // A bit special FrameFilter class : these steps are done inside method go
  // if (!this->next) { return; } // call next filter .. if there is any
  // (this->next)->run(outframe);
}


void SwScaleFrameFilter::go(Frame* frame) { // do the scaling
  if (!this->next) { return; } // exit if no next filter 
  
  if (frame->getFrameClass()!=FrameClass::avbitmap) {
    decoderlogger.log(LogLevel::debug) << "SwScaleFrameFilter: go: ERROR: frame must be AVBitmapFrame " << *frame << std::endl;
    return;
  }
  
  AVBitmapFrame *inputframe = static_cast<AVBitmapFrame*>(frame);
  
  // shorthand
  AVFrame *input_frame  =inputframe->av_frame;
  AVFrame *output_frame =outputframe.av_frame;
  
  if (width != input_frame->width || height != input_frame->height) {
    width  =input_frame->width;
    height =input_frame->height;
    
    if (!sws_ctx) {
    }
    else {
      sws_freeContext(sws_ctx);
    }
    // std::cout << "SwScaleFrameFilter: go: get context: " << std::endl;
    // std::cout << "SwScaleFrameFilter: go: get context: w, h "  << width << " " << height << std::endl;
    sws_ctx =sws_getContext(width, height, (AVPixelFormat)(input_frame->format), target_width, target_height, AV_PIX_FMT_RGB24, SWS_POINT, NULL, NULL, NULL);
  }
  
  sws_scale(sws_ctx, (const uint8_t * const*)input_frame->data, input_frame->linesize, 0, input_frame->height, output_frame->data, output_frame->linesize);
  
  // std::cout << "SwScaleFrameFilter: go: output frame: " << outputframe << std::endl;
  
  if (output_frame->width>0 and output_frame->height>0) {
    outputframe.copyMetaFrom(frame);
    next->run(&outputframe);
  }
} 





