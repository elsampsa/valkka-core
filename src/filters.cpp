/*
 * filters.cpp : Common frame filters.  The FrameFilter base class is defined in frames.h
 * 
 * Copyright 2017 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Valkka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Valkka.  If not, see <http://www.gnu.org/licenses/>. 
 * 
 */

/** 
 *  @file    filters.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
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



TimestampFrameFilter::TimestampFrameFilter(const char* name, FrameFilter* next, long int msdiff_max) : FrameFilter(name,next), msdiff_max(msdiff_max), mstime_delta(0) {
}


// #define TIMESTAMPFILTER_DEBUG
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



RepeatH264ParsFrameFilter::RepeatH264ParsFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next) {
}

void RepeatH264ParsFrameFilter::go(Frame* frame) {// TODO
}




GateFrameFilter::GateFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next), on(false) {
}
  
void GateFrameFilter::go(Frame* frame) {
}
  
void GateFrameFilter::run(Frame* frame) {
  this->go(frame); // manipulate frame
  if (!this->next and on) { return; } // call next filter .. if there is any and if the flag on is set
  (this->next)->run(frame);
}
  
void GateFrameFilter::set() {
  on=true;
}

void GateFrameFilter::unSet() {
  on=false;
}
  

SetSlotFrameFilter::SetSlotFrameFilter(const char* name, FrameFilter* next) : FrameFilter(name,next), n_slot(0) {
}

void SetSlotFrameFilter::go(Frame* frame) {
  if (n_slot>0) {
    frame->n_slot=n_slot;
  }
}
  
void SetSlotFrameFilter::setSlot(SlotNumber n) {
}
  

TimeIntervalFrameFilter::TimeIntervalFrameFilter(const char* name, int mstimedelta, FrameFilter* next) : FrameFilter(name,next), mstimedelta(mstimedelta) { // TODO
}

void TimeIntervalFrameFilter::go(Frame* frame) {
}

