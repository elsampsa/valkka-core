/*
 * file.cpp : File input and output to and from matroska files
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
 *  @file    file.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
 *  
 *  @brief File input and output to and from matroska files
 */ 

#include "file.h"


FileFrameFilter::FileFrameFilter(const char *name, FrameFilter *next) : FrameFilter(name, next), active(false), initialized(false), mstimestamp0(0), zerotimeset(false) {
  int i;
  // two substreams per stream
  contexes.   resize(2,NULL);
  streams.    resize(2,NULL);
  setupframes.resize(2,NULL);
  timebase =av_make_q(1,1000); // we're using milliseconds
}


FileFrameFilter::~FileFrameFilter() {
  deActivate();
}


void FileFrameFilter::go(Frame* frame) {
  std::unique_lock<std::mutex> lk(this->mutex);
  
  // make a copy of the setup frames ..
  if (frame->frametype==FrameType::setup) {
    setupframes[frame->subsession_index]=frame; 
  }
  else if (active) { // everything's ok! just write..
    if (!zerotimeset) {
     mstimestamp0=frame->mstimestamp;
     zerotimeset=true;
    }
    long int dt=(frame->mstimestamp-mstimestamp0);
    if (dt<0) { dt=0; }
  
    filelogger.log(LogLevel::debug) << "FileFrameFilter : writing frame with mstimestamp " << dt << std::endl;
    av_stream=streams[frame->subsession_index];
    
    frame->useAVPacket(dt); // "mirror" data into AVPacket structure with a certain timestamp
    av_interleaved_write_frame(output_context,frame->avpkt);
  }
}
    

bool FileFrameFilter::activate(const char* fname, long int zerotime) {
  std::unique_lock<std::mutex> lk(this->mutex);
  int i;
  AVCodecID codec_id;
  bool initialized = false;
  
  // int avformat_alloc_output_context2 	( 	AVFormatContext **  	ctx, AVOutputFormat *  	oformat, const char *  	format_name, const char *  	filename ) 	
  // int avio_open 	( 	AVIOContext **  	s, const char *  	url, int  	flags ) 	
  // av_warn_unused_result int avformat_write_header 	( 	AVFormatContext *  	s, AVDictionary **  	options ) 	
  
  if (active) {
    deActivate();
  }
  
  // create output context, open files
  i=avformat_alloc_output_context2(&output_context, NULL, "matroska", NULL);
  if (!output_context) {
    filelogger.log(LogLevel::fatal) << "FileFrameFilter : FATAL : could not create output context!  Have you enabled matroska and registered all codecs and muxers? " << std::endl;
    avformat_free_context(output_context);
    exit(2);
  }
  
  i=avio_open(&output_context->pb, fname, AVIO_FLAG_WRITE);  //|| AVIO_SEEKABLE_NORMAL);
  if (i < 0) {
    filelogger.log(LogLevel::fatal) << "Could not open " << fname << std::endl;
    // av_err2str(i)
    avformat_free_context(output_context);
    return false;
  }
    
  // use the saved setup frames (if any) to set up the streams
  Frame *frame; // alias
  for (auto it=setupframes.begin(); it!=setupframes.end(); it++) {
   frame=*it;
   if (!frame) {
   }
   else { // got setupframe
    switch ( (frame->setup_pars).frametype ) { // NEW_CODEC_DEV // when adding new codecs, make changes here: add relevant decoder per codec
      case FrameType::h264: // AV_CODEC_ID_H264
        codec_id =AV_CODEC_ID_H264;
        filelogger.log(LogLevel::debug) << "FileFrameFilter : Initializing H264 at index " << frame->subsession_index << std::endl;
        break;
      case FrameType::pcmu:
        codec_id =AV_CODEC_ID_PCM_MULAW;
        filelogger.log(LogLevel::debug) << "FileFrameFilter : Initializing PCMU at index " << frame->subsession_index << std::endl;
        break;
      default:
        codec_id=AV_CODEC_ID_NONE;
        filelogger.log(LogLevel::debug) << "FileFrameFilter : Could not init subsession " << frame->subsession_index << std::endl;
        break;
      }
      // AVCodecContext* avcodec_alloc_context3(const AVCodec * codec)
      // AVCodec* avcodec_find_decoder(enum AVCodecID id)
      // AVStream* avformat_new_stream(AVFormatContext * s, const AVCodec * c )
      // int avcodec_parameters_from_context(AVCodecParameters * par, const AVCodecContext *  codec)
    
      if (codec_id!=AV_CODEC_ID_NONE) {
        av_codec_context =avcodec_alloc_context3(avcodec_find_decoder(codec_id));
        av_codec_context->width =BitmapPars::N720::w; // dummy values .. otherwise mkv muxer refuses to co-operate
        av_codec_context->height=BitmapPars::N720::h;
        
        av_stream        =avformat_new_stream(output_context,av_codec_context->codec); // av_codec_context->codec == AVCodec (i.e. we create a stream having a certain codec)
        av_stream->time_base=timebase;
        
        i=avcodec_parameters_from_context(av_stream->codecpar,av_codec_context);
        
        contexes[frame->subsession_index] =av_codec_context;
        streams [frame->subsession_index] =av_stream;
        
        initialized =true; // so, at least one substream init'd
      }
    } // got setupframe
  }
  
  if (!initialized) {
    return false;
  }
  
  i=avformat_write_header(output_context, NULL);
  if (i < 0) {
    filelogger.log(LogLevel::fatal) << "Error occurred when opening output file " << fname << std::endl;
    // av_err2str(i)
    avformat_free_context(output_context);
    return false;
  }
  
  // so far so good ..
  active       =true;
  // mstimestamp0 =getCurrentMsTimestamp(); // nopes
  if (zerotime>0) { // user wants to set time reference explicitly and not from first arrived packet ..
    mstimestamp0=zerotime;
    zerotimeset=true;
  }
  else {
    zerotimeset=false;
  }
  filename     =std::string(fname);
  return true;
}  

  

void FileFrameFilter::deActivate() {
  std::unique_lock<std::mutex> lk(this->mutex);
  int i;
  
  if (active) {
    av_write_trailer(output_context);
    for(auto it=contexes.begin(); it!=contexes.end(); ++it) {
      i=avcodec_close(*it);
    }
    avio_closep(&output_context->pb);
    active=false;
    avformat_free_context(output_context);
  }
}
  

    

    