/*
 * frames.cpp : Valkka frame type implementations
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
 *  @file    frames.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.5 
 *  
 *  @brief Basic frame types implementations
 *
 *  @section DESCRIPTION
 *  
 *  Yes, the description
 *
 */
 
#include "frames.h"
#include "logging.h"



std::ostream &operator<<(std::ostream &os, SetupPars const &m) {
 // return os << "Setup: subsession_index="<<m.subsession_index<< " frametype="<< int(m.frametype);
 return os << "Setup: frametype="<< int(m.frametype);
}

std::ostream &operator<<(std::ostream &os, AVFramePars const &m) {
  return os << "AVFrame: ";
}

std::ostream &operator<<(std::ostream &os, YUVFramePars const &m) {
  return os << "YUVFrame: bmtype="<<m.bmtype;
}

std::ostream &operator<<(std::ostream &os, H264Pars const &m) {
 return os << "H264: slice_type="<<m.slice_type;
}

std::ostream &operator<<(std::ostream &os, PCMUPars const &m) {
 return os << "PCMU";
}


// eh.. we could do a std::map as well..
FrameType codec_id_to_frametype(AVCodecID av_codec_id) {
  switch (av_codec_id) {
    case AV_CODEC_ID_H264:
      return FrameType::h264;
      break;
    case AV_CODEC_ID_PCM_MULAW:
      FrameType::pcmu;
      break;
    default:
      FrameType::none;
      break;
  }
}


AVCodecID frametype_to_codec_id(FrameType frametype) {
  switch (frametype) {
    case FrameType::h264:
      return AV_CODEC_ID_H264;
      break;
    case FrameType::pcmu:
      return AV_CODEC_ID_PCM_MULAW;
      break;
    default:
      return AV_CODEC_ID_NONE;
      break;
  }
}



// Frame::Frame() : mstimestamp(0), rel_mstimestamp(0), n_slot(0), frametype(FrameType::none), subsession_index(0), av_frame(NULL), av_codec_context(NULL), yuvpbo(NULL) {
Frame::Frame() : mstimestamp(0), n_slot(0), frametype(FrameType::none), subsession_index(0), av_frame(NULL), av_codec_context(NULL), yuvpbo(NULL), avpkt(NULL) {
}


Frame::~Frame() {
  if (!avpkt) {
  }
  else {
    av_free_packet(avpkt);
  }
}


void Frame::setMsTimestamp (long int t) {
  mstimestamp=t;
}


void Frame::copyMeta(Frame* f) {
  /* f->frametype    =frametype; // not this..!  We want to copy metadata typically between frames of different type.. (say avframe => yuvframe)
  f->setup_pars  =setup_pars;     
  f->h264_pars   =h264_pars;  
  f->pcmu_pars   =pcmu_pars;                     
  f->av_pars     =av_pars;                     
  f->yuv_pars    =yuv_pars;
  */
  f->mstimestamp      =mstimestamp;                 
  // f->rel_mstimestamp  =rel_mstimestamp;
  f->subsession_index =subsession_index;
  f->n_slot           =n_slot;
}


/*
void Frame::calcRelMsTimestamp (struct timeval time) {
  // relative timestamp = (abs mstimestamp - current mstime)
  rel_mstimestamp=mstimestamp-(time.tv_sec*1000+time.tv_usec/1000);
}
*/

 
long int Frame::getMsTimestamp() {
  return mstimestamp;
}


/*
long int Frame::getRelMsTimestamp() {
  return rel_mstimestamp;
}
*/


void Frame::reserve(std::size_t n_bytes) {
  this->payload.reserve(n_bytes);
}


void Frame::resize(std::size_t n_bytes) {
  this->payload.resize(n_bytes,0);
}

/* // so, don't touch av_frame, av_codec_context or yuvpbo here at all.  They are not managed by Frame instances.
void Frame::release() {
  if (yuvpbo!=NULL) {
    delete yuvpbo; // this, of course, always inside the thread that controls opengl.  The YUVPBO instance is managed by Frame
    yuvpbo=NULL;
  }
  // don't release av_frame here .. it is merely a pointer to an av_frame instance that is managed by a Decoder
  if (av_frame!=NULL) {
    av_frame_free(&av_frame);
    av_frame=NULL;
  }
}
*/


void Frame::reset() {
  frametype=FrameType::none;
  yuvpbo           =NULL;
  av_frame         =NULL;
  av_codec_context =NULL;
  mstimestamp=0;
  // rel_mstimestamp=0;
}


std::string Frame::dumpPayload() {
  std::stringstream tmp;
  
  for(std::vector<uint8_t>::iterator it=payload.begin(); it<min(payload.end(),payload.begin()+20); ++it) {
    tmp << int(*(it)) <<" ";
  }
  
  return tmp.str();
}

std::string Frame::dumpAVFrame() {
  std::stringstream tmp;
  
  if (!av_frame) {
  }
  else {
    tmp << "AVFrame height         : "<< av_frame->height<< std::endl;
    tmp << "AVFrame width          : "<< av_frame->width << std::endl;
    tmp << "AVFrame linesizes      : "<< av_frame->linesize[0] << " " << av_frame->linesize[1] << " " << av_frame->linesize[2] << std::endl;
    tmp << "AVFrame format         : "<< av_frame->format << std::endl; // AV_PIX_FMT_YUV420P == 0
  }
  if (!av_codec_context) {
  }
  else {
    tmp << "AVCodecContext pix_fmt : "<< av_codec_context->pix_fmt << std::endl;
  }
  return tmp.str();
}


void Frame::fillPars() {
  switch(frametype) {
    case FrameType::h264:
      if (payload.size()>4) { 
        h264_pars.slice_type = ( payload[4] & 31 );
      }
      break;
    default:
      break;
  }
}


void Frame::useAVPacket(long int pts) {
  if (!avpkt) {
    avpkt= new AVPacket();
    av_init_packet(avpkt);
  }
  
  avpkt->data =payload.data();
  avpkt->size =payload.size();
  avpkt->stream_index=subsession_index;
  
  if (frametype==FrameType::h264 and h264_pars.slice_type==H264SliceType::sps) { // we assume that frames always come in the following sequence: sps, pps, i, etc.
    avpkt->flags=AV_PKT_FLAG_KEY;
  }
  
  // std::cout << "Frame : useAVPacket : pts =" << pts << std::endl;
  
  if (pts>=0) {
    avpkt->pts=(int64_t)pts;
  }
  else {
    avpkt->pts=AV_NOPTS_VALUE;
  }
  
  // std::cout << "Frame : useAVPacket : final pts =" << pts << std::endl;
  
  avpkt->dts=AV_NOPTS_VALUE; // let muxer set it automagically
}


void Frame::reportMsTime() {
 std::cout << "Frame : reportMsTime : timediff to current time : " << mstimestamp-getCurrentMsTimestamp() << std::endl; // i.e. positive: in the future, negative: in the past
}


void Frame::fromAVPacket(AVPacket *pkt) {
  payload.resize(pkt->size);
  memcpy(payload.data(),pkt->data,pkt->size);
  // TODO: optimally, this would be done only once - in copy-on-write when writing to fifo, at the thread border
  subsession_index=pkt->stream_index;
  // frametype=FrameType::h264; // not here .. avpkt carries no information about the codec
  mstimestamp=(long int)pkt->pts;
}


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
}
  
void DummyFrameFilter::go(Frame* frame) { 
  if (verbose) {
    std::cout << "DummyFrameFilter : "<< this->name << " : got frame : " << *(frame) << std::endl;
  }
}


