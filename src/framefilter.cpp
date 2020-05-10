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
 *  @version 0.17.4 
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
  std::cout << "BriefInfoFrameFilter : "<< this->name << " : " << *(frame) << " dT=" << frame->mstimestamp-getCurrentMsTimestamp() << std::endl;
}


ThreadSafeFrameFilter::ThreadSafeFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next) {
}

void ThreadSafeFrameFilter::run(Frame* frame) {
    if (!this->next) { return; } // call next filter .. if there is any
    {
        std::unique_lock<std::mutex> lk(this->mutex); 
        (this->next)->run(frame);
    }
}

void ThreadSafeFrameFilter::go(Frame* frame) {
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



ForkFrameFilterN::ForkFrameFilterN(const char* name) : FrameFilter(name, NULL) {
}


ForkFrameFilterN::~ForkFrameFilterN() {
}


void ForkFrameFilterN::run(Frame* frame) {
  std::unique_lock<std::mutex> lk(mutex); // mutex protected .. so that if user is adding new terminals, this wont crash
  for (auto it=framefilters.begin(); it!=framefilters.end(); it++) {
    it->second->run(frame);
  }
}


bool ForkFrameFilterN::connect(const char* tag, FrameFilter* filter) {
  // std::map<std::string,FrameFilter*> framefilters;
  std::unique_lock<std::mutex> lk(mutex);
  std::string nametag(tag);
  
  auto it=framefilters.find(nametag);
  
  if (it!=framefilters.end()) {
    filterlogger.log(LogLevel::fatal) << "ForkFrameFilterN : connect : key "<<  nametag << " already used " << std::endl;
    return false;
  }
  
  framefilters.insert(it, std::pair<std::string,FrameFilter*>(nametag,filter));
  return true;
}


bool ForkFrameFilterN::disconnect(const char* tag) {
  std::unique_lock<std::mutex> lk(mutex);
  std::string nametag(tag);
  
  auto it=framefilters.find(nametag);
  
  if (it==framefilters.end()) {
    filterlogger.log(LogLevel::fatal) << "ForkFrameFilterN : disconnect : key "<<  nametag << " does not exist " << std::endl;
    return false;
  }
  
  framefilters.erase(it);
  return true;
}


void ForkFrameFilterN::go(Frame* frame) { // dummy virtual function
}



SlotFrameFilter::SlotFrameFilter(const char* name, SlotNumber n_slot, FrameFilter* next) : FrameFilter(name,next), n_slot(n_slot) {
}
    

void SlotFrameFilter::go(Frame* frame) {
  frame->n_slot=n_slot;
}



PassSlotFrameFilter::PassSlotFrameFilter(const char* name, SlotNumber n_slot, FrameFilter* next) : FrameFilter(name,next), n_slot(n_slot) {
}
    
void PassSlotFrameFilter::go(Frame* frame) {
}

void PassSlotFrameFilter::run(Frame* frame) {
    if (!next) { return; }
    if (frame->n_slot == n_slot) {
        next->run(frame);
    }
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
  
  // std::cerr << "BufferSource: IN0: " << *frame << std::endl;
  
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
  
  // std::cerr << "BufferSource: IN0: " << *frame << std::endl;
  
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



RepeatH264ParsFrameFilter::RepeatH264ParsFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next), sps(), pps(), phase(-1) {
}

void RepeatH264ParsFrameFilter::go(Frame* frame) {
}

// #define repeat_ff_verbose 1

void RepeatH264ParsFrameFilter::run(Frame* frame) {
    if (!next) {return;}
    
    if (frame->getFrameClass() != FrameClass::basic) { // all other frames than BasicFrame, just pass-through
        (this->next)->run(frame);
    }
    else {
        
#ifdef repeat_ff_verbose
        std::cout << ">>> RepeatH264ParsFrameFilter: got frame" << std::endl;
#endif
        
        BasicFrame* basic_frame = static_cast<BasicFrame*>(frame);
        
        if (basic_frame->codec_id == AV_CODEC_ID_H264) {
            // H264SliceType::sps, pps, i
            unsigned slice_type = basic_frame->h264_pars.slice_type;
            if      (slice_type == H264SliceType::sps) {    // SPS
                sps.copyFrom(basic_frame); // cache sps
#ifdef repeat_ff_verbose
                std::cout << ">>> RepeatH264ParsFrameFilter: cached sps" << std::endl;
#endif
                phase = 0;                
            }                                               // SPS
            
            else if (slice_type == H264SliceType::pps) {    // PPS
                pps.copyFrom(basic_frame); // cache pps
#ifdef repeat_ff_verbose
                std::cout << ">>> RepeatH264ParsFrameFilter: cached pps" << std::endl;
#endif
                if (phase == 0) { // so, last packet was sps
                    phase = 1;
                }
                else {
                    phase = -1;
                }
            }                                               // PPS
            
            else if (slice_type == H264SliceType::i) {      // KEY
                if (phase == 1) { // all fine - the packets came in the right order: sps, pps and now i-frame
                }
                else {
                    filterlogger.log(LogLevel::debug) << "RepeatH264ParsFrameFilter: re-sending sps & pps" << std::endl;
                    
                    if ( (sps.codec_id != AV_CODEC_ID_NONE) and (pps.codec_id != AV_CODEC_ID_NONE) ) { // so, these have been cached correctly
                        sps.mstimestamp = frame->mstimestamp;
                        pps.mstimestamp = frame->mstimestamp;
                        (this->next)->run((Frame*)(&sps));
                        (this->next)->run((Frame*)(&pps));
                    }
                    else {
                        filterlogger.log(LogLevel::fatal) << "RepeatH264ParsFrameFilter: re-sending sps & pps required but they're n/a" << std::endl;
                    }
                }
                phase = -1;
            }                                               // KEY
            
            (this->next)->run(frame); // send the original frame
            
#ifdef repeat_ff_verbose
            std::cout << ">>> RepeatH264ParsFrameFilter: phase=" << phase << std::endl;
#endif
            
        }
        else { // just passthrough
            (this->next)->run(frame);
        }
    }
}




GateFrameFilter::GateFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next), on(false), config_frames(true) {
}
  
void GateFrameFilter::go(Frame* frame) {
}
  
void GateFrameFilter::run(Frame* frame) {
  std::unique_lock<std::mutex> lk(mutex);
  // this->go(frame); // manipulate frame
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


SwitchFrameFilter::SwitchFrameFilter(const char* name, FrameFilter* next1, FrameFilter* next2) : FrameFilter(name, NULL), next1(next1), next2(next2), index(1) {
}

void SwitchFrameFilter::go(Frame* frame) {
}

void SwitchFrameFilter::run(Frame* frame) {
    std::unique_lock<std::mutex> lk(mutex);
    // this->go(frame); // manipulate frame
    if (index==1 and next1) {
        next1->run(frame);
    }
    else if (index==2 and next2) {
        next2->run(frame);
    }
}
    
void SwitchFrameFilter::set1() {
    std::unique_lock<std::mutex> lk(mutex);
    index=1;
}

void SwitchFrameFilter::set2() {
    std::unique_lock<std::mutex> lk(mutex);
    index=2;
}


TypeFrameFilter::TypeFrameFilter(const char* name, FrameClass frameclass, FrameFilter* next) : FrameFilter(name, next), frameclass(frameclass) {
}

void TypeFrameFilter::go(Frame* frame) {
}


void TypeFrameFilter::run(Frame* frame) {
    if (next and frame->getFrameClass()==frameclass) {
        next->run(frame);
    }
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
        setupframe = *(static_cast<SetupFrame*>(frame)); // create a cached copy of the SetupFrame
        if (setupframe.sub_type == SetupFrameType::stream_init) { // TODO: shouldn't we have an array of setupframes (for each substream)
            got_setup=true;
        }
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

// #define INTERVAL_VERBOSE = 1

void TimeIntervalFrameFilter::run(Frame* frame) {

    #ifdef INTERVAL_VERBOSE
        std::cout << std::endl << "TimeIntervalFrameFilter: " << std::endl;
    #endif
    if (!this->next) { return; } // calls next filter .. if there is any
    #ifdef INTERVAL_VERBOSE
        std::cout << std::endl << "TimeIntervalFrameFilter: mstimestamps=" << frame->mstimestamp << " " << prevmstimestamp << std::endl;
        std::cout << std::endl << "TimeIntervalFrameFilter: delta       =" << (frame->mstimestamp-prevmstimestamp) << std::endl;
    #endif
    if ( (frame->mstimestamp-prevmstimestamp)>=mstimedelta ) {
        #ifdef INTERVAL_VERBOSE
            std::cout << std::endl << "TimeIntervalFrameFilter: WRITE" << std::endl;
        #endif
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
  AVFrame *output_frame =outputframe.av_frame; // AVRGBFrame => AVBitmapFrame => AVMediaFrame constructor has reserved av_frame member
  
  int nb = av_image_alloc(output_frame->data, output_frame->linesize, target_width, target_height, AV_PIX_FMT_RGB24, ALIGNMENT); // ALIGNMENT at constant.h
  if (nb <= 0) {
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
  av_freep(&((outputframe.av_frame)->data[0]));
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
  // NOTE: input_frame's are YUV, coming from the decoder.  They are aligned.  We can then decide to scrap the alignment here .. in any case, we must do it at some moment
  // before passing RGB frames further downstream (to analyzers etc.)
  
  // std::cout << "SwScaleFrameFilter: go: output frame: " << outputframe << std::endl;
  
  if (output_frame->width>0 and output_frame->height>0) {
    outputframe.copyMetaFrom(frame);
    next->run(&outputframe);
  }
}





