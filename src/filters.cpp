/*
 * filters.cpp : Common frame filters.  The FrameFilter base class is defined in frames.h
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
 *  @file    filters.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.0 
 *  
 *  @brief Common frame filters.  The FrameFilter base class is defined in frames.h
 *
 *  @section DESCRIPTION
 *
 */ 

#include "filters.h"
#include "logging.h"


InfoFrameFilter::InfoFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next) {
}

void InfoFrameFilter::go(Frame* frame) {
  std::cout << "InfoFrameFilter: " << name << " start dump>> " <<std::endl;
  std::cout << "InfoFrameFilter: FRAME   : "<< *(frame) << std::endl;
  std::cout << "InfoFrameFilter: PAYLOAD : [";
  std::cout << frame->dumpPayload();
  std::cout << "]" << std::endl;
  std::cout << "InfoFrameFilter:<" << frame->dumpAVFrame() << ">" << std::endl;
  std::cout << "InfoFrameFilter: timediff: " << frame->mstimestamp-getCurrentMsTimestamp() << std::endl;
  std::cout << "InfoFrameFilter: " << name << " <<end dump   " <<std::endl;
  
}


BriefInfoFrameFilter::BriefInfoFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next) {
}

void BriefInfoFrameFilter::go(Frame* frame) {
  std::cout << "DummyFrameFilter : "<< this->name << " : got frame : " << *(frame) << " dT=" << frame->getMsTimestamp()-getCurrentMsTimestamp() << std::endl;
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
 filterlogger.log(LogLevel::debug) << "ForkFrameFilter : "<< this->name << " : got frame : " << *(frame) << std::endl;
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
  
  std::copy((frame->payload).begin(), (frame->payload).end(), std::ostreambuf_iterator<char>(fout));
  
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
  
  frame->setMsTimestamp(frame->mstimestamp+mstime_delta);
#ifdef TIMESTAMPFILTER_DEBUG
  std::cout << "TimestampFrameFilter: go: final frame->mstimestamp " << frame->mstimestamp << std::endl;
#endif
}



TimestampFrameFilter2::TimestampFrameFilter2(const char* name, FrameFilter* next, long int msdiff_max) : FrameFilter(name,next), msdiff_max(msdiff_max), mstime_delta(0), savedtimestamp(0) {
}


void TimestampFrameFilter2::go(Frame* frame) {
  long int ctime, corrected, diff;
  
  if ( (frame->mstimestamp-savedtimestamp)>600000 ) {
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
  
  frame->setMsTimestamp(frame->mstimestamp+mstime_delta);
#ifdef TIMESTAMPFILTER_DEBUG
  std::cout << "TimestampFrameFilter2: go: final frame->mstimestamp " << frame->mstimestamp << std::endl;
#endif
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
  else if (frame->frametype==FrameType::setup and config_frames) { // .. if flag is not set, pass still config frames if config_frames is set
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


SwScaleFrameFilter::SwScaleFrameFilter(const char* name, int target_width, int target_height, FrameFilter* next) : FrameFilter(name,next), target_width(target_width), target_height(target_height), outframe(Frame()), sws_ctx(NULL) {
  setTargetFmt();
  outframe.frametype=FrameType::avframe;
  outframe.av_frame=av_frame_alloc();
  
  // output AVFrame alias
  AVFrame *out_avframe  = outframe.av_frame;
  
  out_avframe->width=target_width;
  out_avframe->height=target_height;
  
  int nb;
  nb =av_image_alloc(out_avframe->data, out_avframe->linesize, out_avframe->width, out_avframe->height, target_pix_fmt, 1);
  if (nb>0) {
    decoderlogger.log(LogLevel::debug) << "SwScaleFrameFilter: constructor: reserved " << nb << " bytes for the bitmap" << std::endl;
  }
  else {
    // decoderlogger.log(LogLevel::fatal) << "SwScaleFrameFilter: constructor: FATAL: could not reserve image " << std::endl;
    std::perror("SwScaleFrameFilter: constructor: FATAL: could not reserve image");
    exit(2);
  }
}


SwScaleFrameFilter::~SwScaleFrameFilter() {
  av_freep(outframe.av_frame->data);
  av_frame_free(&(outframe.av_frame));
  if (!sws_ctx) {
  }
  else {
    sws_freeContext(sws_ctx);
  }
}


void SwScaleFrameFilter::setTargetFmt() {
  target_pix_fmt=AV_PIX_FMT_RGB24;
}


void SwScaleFrameFilter::run(Frame* frame) { // AVThread calls this ..
  // std::cout << "SwScaleFrameFilter: run" << std::endl;
  this->go(frame); // manipulate frame - in this case, scale from yuv to rgb
  // A bit special FrameFilter class : these steps are done inside method go
  // if (!this->next) { return; } // call next filter .. if there is any
  // (this->next)->run(outframe);
}


void SwScaleFrameFilter::go(Frame* frame) { // do the scaling
  if (frame->frametype==FrameType::avframe) {
    if (frame->av_codec_context->codec_type==AVMEDIA_TYPE_VIDEO) {// VIDEO
      /*
      if ( // ALLOWED PIXEL FORMATS // NEW_CODEC_DEV: is your pixel format supported?
        frame->av_codec_context->pix_fmt==  AV_PIX_FMT_YUV420P  ||
        frame->av_codec_context->pix_fmt==  AV_PIX_FMT_YUVJ420P
      ) 
      */
    
        
      // input AVFrame aliases
      AVFrame         *in_avframe   = frame->av_frame;
      AVCodecContext  *ctx          = frame->av_codec_context;
      // output AVFrame alias
      AVFrame         *out_avframe  = outframe.av_frame;
      
      if (!sws_ctx) {
      }
      /*
      else { // so, scaling context has been set .. let's see if input dimensions still hold
        if (in_avframe->width!=sws_ctx->dstW or in_avframe->height!=sws_ctx->dstH) { // dimensions changed - time to reinit
          sws_freeContext(sws_ctx);
          sws_ctx=NULL;
        }
      }
      */
      
      if (!sws_ctx) { // got frame for the first time
        sws_ctx =sws_getContext(in_avframe->width, in_avframe->height, ctx->pix_fmt, out_avframe->width, out_avframe->height, target_pix_fmt, SWS_POINT, NULL, NULL, NULL);
      }
        
      // refer to: https://ffmpeg.org/doxygen/trunk/group__libsws.html#gae531c9754c9205d90ad6800015046d74
      sws_scale(sws_ctx, (const uint8_t * const*)in_avframe->data, in_avframe->linesize, 0, in_avframe->height, out_avframe->data, out_avframe->linesize);
      
      if (!this->next) { return; } // call next filter .. if there is any
      
      frame->copyMeta(&outframe); // mstimestamp, subsession index, slot, etc.
      
      // copy data to the actual payload..
      // std::cout << "SwScaleFrameFilter: linesizes: " << out_avframe->linesize[0] << " " << out_avframe->linesize[1] << " " << out_avframe->linesize[2] << std::endl; // linesize[0] == width * 3 (for rgb)
      std::size_t n=out_avframe->linesize[0]*out_avframe->height;
      outframe.payload.resize(n);
      memcpy(outframe.payload.data(),out_avframe->data[0], n);
      
    
      //outframe.payload.resize(out_avframe->linesize
      //outframe->data
      
      
      (this->next)->run(&outframe);
    } // VIDEO
    else {
      decoderlogger.log(LogLevel::debug) << "SwScaleFrameFilter: go: got avframe that's not bitmap: " << *frame << std::endl;
    }
    
  }
  else {
    // decoderlogger.log(LogLevel::fatal) << "SwScaleFrameFilter: go: FATAL: needs a Frame with frame->frametype = FrameType::avframe. Got " << *frame << std::endl;
  }
}


