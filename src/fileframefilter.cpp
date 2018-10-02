/*
 * fileframefilter.cpp : File input to matroska files
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
 *  @version 0.6.0 
 *  
 *  @brief File input to matroska files
 */ 

#include "fileframefilter.h"


FileFrameFilter::FileFrameFilter(const char *name, FrameFilter *next) : FrameFilter(name, next), active(false), initialized(false), mstimestamp0(0), zerotimeset(false), ready(false) {
  int i;
  // two substreams per stream
  contexes.   resize(2,NULL);
  streams.    resize(2,NULL);
  // setupframes.resize(2,NULL);
  setupframes.resize(2);
  timebase =av_make_q(1,1000); // we're using milliseconds
  avpkt    =new AVPacket();
  av_init_packet(avpkt);  
}


FileFrameFilter::~FileFrameFilter() {
  Frame *frame;
  deActivate();
  av_free_packet(avpkt);
  delete avpkt;
  /*
  for (auto it=setupframes.begin(); it!=setupframes.end(); it++) {
    frame=*it;
    if (frame!=NULL) {
      delete frame;
    }
  }
  */
}


void FileFrameFilter::go(Frame* frame) {
  std::unique_lock<std::mutex> lk(this->mutex);
  
  // make a copy of the setup frames ..
  if (frame->getFrameClass()==FrameClass::setup) {    
    SetupFrame *setupframe = static_cast<SetupFrame*>(frame);

    if (setupframe->subsession_index>1) {
      filelogger.log(LogLevel::fatal) << "FileFrameFilter : too many subsessions! " << std::endl;
    }
    else {
      filelogger.log(LogLevel::debug) << "FileFrameFilter :  go : got setup frame " << *setupframe << std::endl;
      setupframes[setupframe->subsession_index].copyFrom(setupframe);
    }
    return;
  }
  
  else if (frame->getFrameClass()==FrameClass::basic) {
    BasicFrame *basicframe =static_cast<BasicFrame*>(frame);
    
    if (!ready) {
      if (setupframes[0].subsession_index > -1) { // we have got at least one setupframe and after that, payload
        ready=true;
      }
    }
    
    if (ready and active and !initialized) { // got setup frames, writing has been requested, but file has not been opened yet
      initFile(); // modifies member initialized
      if (!initialized) { // can't init this file.. de-activate
        deActivate_();
      }
    }
    
    if (initialized) { // everything's ok! just write..
      if (!zerotimeset) {
        mstimestamp0=basicframe->mstimestamp;
        zerotimeset=true;
      }
      long int dt=(basicframe->mstimestamp-mstimestamp0);
      if (dt<0) { dt=0; }
    
      filelogger.log(LogLevel::debug) << "FileFrameFilter : writing frame with mstimestamp " << dt << std::endl;
      //av_stream=streams[frame->subsession_index];
      
      internal_frame.copyFrom(basicframe);
      internal_frame.mstimestamp=dt;
      internal_frame.fillAVPacket(avpkt);
      av_interleaved_write_frame(output_context,avpkt);
    }
    else {
      // std::cout << "FileFrameFilter: go: discarding frame" << std::endl;
    }
  } // BasicFrame
  else { // don't know how to handle that frame ..
  }
}

    
void FileFrameFilter::initFile() {
  int i;
  AVCodecID codec_id;
  initialized = false;
  
  // int avformat_alloc_output_context2 	( 	AVFormatContext **  	ctx, AVOutputFormat *  	oformat, const char *  	format_name, const char *  	filename ) 	
  // int avio_open 	( 	AVIOContext **  	s, const char *  	url, int  	flags ) 	
  // av_warn_unused_result int avformat_write_header 	( 	AVFormatContext *  	s, AVDictionary **  	options ) 	
  
  // create output context, open files
  i=avformat_alloc_output_context2(&output_context, NULL, "matroska", NULL);
  if (!output_context) {
    filelogger.log(LogLevel::fatal) << "FileFrameFilter : initFile : FATAL : could not create output context!  Have you enabled matroska and registered all codecs and muxers? " << std::endl;
    avformat_free_context(output_context);
    exit(2);
  }
  
  i=avio_open(&output_context->pb, filename.c_str(), AVIO_FLAG_WRITE);  //|| AVIO_SEEKABLE_NORMAL);
  if (i < 0) {
    filelogger.log(LogLevel::fatal) << "FileFrameFilter : initFile : could not open " << filename << std::endl;
    // av_err2str(i)
    avformat_free_context(output_context);
    return;
  }
  
  // use the saved setup frames (if any) to set up the streams
  // Frame *frame; // alias
  for (auto it=setupframes.begin(); it!=setupframes.end(); it++) {
   SetupFrame &setupframe=*it;
   if (setupframe.subsession_index<0) { // not been initialized
   }
   else { // got setupframe
    AVCodecID codec_id =setupframe.codec_id;
     
    // AVCodecContext* avcodec_alloc_context3(const AVCodec * codec)
    // AVCodec* avcodec_find_decoder(enum AVCodecID id)
    // AVStream* avformat_new_stream(AVFormatContext * s, const AVCodec * c )
    // int avcodec_parameters_from_context(AVCodecParameters * par, const AVCodecContext *  codec)
    if (codec_id!=AV_CODEC_ID_NONE) {
      AVCodecContext *av_codec_context;
      AVStream       *av_stream;
      
      av_codec_context =avcodec_alloc_context3(avcodec_find_decoder(codec_id));
      av_codec_context->width =N720.width; // dummy values .. otherwise mkv muxer refuses to co-operate
      av_codec_context->height=N720.height;
      
      av_stream        =avformat_new_stream(output_context,av_codec_context->codec); // av_codec_context->codec == AVCodec (i.e. we create a stream having a certain codec)
      av_stream->time_base=timebase;
      
      i=avcodec_parameters_from_context(av_stream->codecpar,av_codec_context);
      
      // std::cout << "FileFrameFilter : initFile : context and stream " << std::endl;
      contexes[setupframe.subsession_index] =av_codec_context;
      streams [setupframe.subsession_index] =av_stream;
      initialized =true; // so, at least one substream init'd
    }
   } // got setupframe
  }
  
  if (!initialized) {
    return;
  }
  
  // contexes, streams, output_context reserved !
  i=avformat_write_header(output_context, NULL);
  if (i < 0) {
    filelogger.log(LogLevel::fatal) << "FileFrameFilter :  initFile : Error occurred when opening output file " << filename << std::endl;
    // av_err2str(i)
    closeFile();
    return;
  }
  
  // so far so good ..
  if (zerotime>0) { // user wants to set time reference explicitly and not from first arrived packet ..
    mstimestamp0=zerotime;
    zerotimeset=true;
  }
  else {
    zerotimeset=false;
  }
  
}


void FileFrameFilter::closeFile() {
 if (initialized) {
   // std::cout << "FileFrameFilter: closeFile" << std::endl;
   avio_closep(&output_context->pb);
   avformat_free_context(output_context);
   for (auto it=contexes.begin(); it!=contexes.end(); ++it) {
      if (*it!=NULL) {
        // std::cout << "FileFrameFilter: closeFile: context " << (long unsigned)(*it) << std::endl;
        avcodec_close(*it);
        avcodec_free_context(&*it);
       *it=NULL;
      }
  }
  for (auto it=streams.begin(); it!=streams.end(); ++it) {
    if (*it!=NULL) {
      // std::cout << "FileFrameFilter: closeFile: stream" << std::endl;
      // eh.. nothing needed here.. enough to call close on the context
      *it=NULL;
    }
  }
 }
 initialized=false;
}


void FileFrameFilter::deActivate_() {
  if (initialized) {
    av_write_trailer(output_context);
    closeFile();
  }
  
  active=false;
}

    
void FileFrameFilter::activate(const char* fname, long int zerotime) {
  std::unique_lock<std::mutex> lk(this->mutex);
  if (active) {
    deActivate_();
  }
  
  filelogger.log(LogLevel::debug) << "FileFrameFilter :  activate : requested for " << fname << std::endl;
  this->filename  =std::string(fname);
  this->zerotime  =zerotime;
  this->active    =true;
}  

  
void FileFrameFilter::deActivate() {
  std::unique_lock<std::mutex> lk(this->mutex);
  
  // std::cout << "FileFrameFilter: deActivate:" << std::endl;
  deActivate_();
  // std::cout << "FileFrameFilter: deActivate: bye" << std::endl;
}
  

    

    