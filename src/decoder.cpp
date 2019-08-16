/*
 * decoders.cpp : FFmpeg decoders
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
 *  @file    decoders.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.13.2 
 *  @brief   FFmpeg decoders
 */ 

#include "decoder.h"
#include "logging.h"

// https://stackoverflow.com/questions/14914462/ffmpeg-memory-leak
// #define AV_REALLOC

Decoder::Decoder() : has_frame(false) {
};


Decoder::~Decoder() {
};


long int Decoder::getMsTimestamp() {
  return in_frame.mstimestamp;
}


void Decoder::input(Frame *f) {
  if (f->getFrameClass()!=FrameClass::basic) {
    decoderlogger.log(LogLevel::normal) << "Decoder: input: can only accept BasicFrame" << std::endl;
  }
  else {
    in_frame.copyFrom(f);
  }
}

bool Decoder::hasFrame() {
    return has_frame;
}


Frame* DummyDecoder::output() {
  return &out_frame;
}


void DummyDecoder::flush() {
};


bool DummyDecoder::pull() {
  out_frame=in_frame;
  has_frame = true;
  return true;
}



AVDecoder::AVDecoder(AVCodecID av_codec_id, int n_threads) : Decoder(), av_codec_id(av_codec_id), n_threads(n_threads) {
    int retcode;

    has_frame = false;
    
    // av_packet       =av_packet_alloc(); // oops.. not part of the public API (defined in avcodec.c instead of avcodec.h)
    av_packet       =new AVPacket();
    av_init_packet(av_packet);
    // av_frame        =av_frame_alloc();
    
    // std::cout << "AVDecoder : AV_CODEC_ID=" << av_codec_id << " AV_CODEC_ID_H264=" << AV_CODEC_ID_H264 << " AV_CODEC_ID_INDEO3=" << AV_CODEC_ID_INDEO3 <<std::endl;
    
    av_codec        =avcodec_find_decoder(av_codec_id);
    if (!av_codec) {
            std::perror("AVDecoder: FATAL: could not find av_codec");
            exit(2);
    }
    av_codec_context = avcodec_alloc_context3(av_codec);
    
    ///*
    av_codec_context->thread_count = this->n_threads;
    if (this->n_threads > 1) {
        std::cout << "AVDecoder: using multithreading with : " << this->n_threads << std::endl;
        decoderlogger.log(LogLevel::debug) << "AVDecoder: using multithreading with " << this->n_threads << std::endl;
        // av_codec_context->thread_type = FF_THREAD_SLICE; // decode different parts of the frame in parallel // FF_THREAD_FRAME = decode several frames in parallel
        // woops .. I get "data partition not implemented" for this version of ffmpeg (3.4)
        av_codec_context->thread_type = FF_THREAD_FRAME; // now frames come in bursts .. think about this
    }
    else {
        av_codec_context->thread_type = FF_THREAD_FRAME;
    }
    //*/
    
    // av_codec_context->refcounted_frames = 1; // monitored the times av_buffer_alloc was called .. makes no different really
    retcode = avcodec_open2(av_codec_context, av_codec, NULL);
    
    // is this going to change at some moment .. ? https://blogs.gentoo.org/lu_zero/2016/03/29/new-avcodec-api/
    // aha..! it's here: https://ffmpeg.org/doxygen/3.2/group__lavc__decoding.html#ga58bc4bf1e0ac59e27362597e467efff3
    if (retcode!=0) {
            std::perror("AVDecoder: FATAL could not open codec");
            exit(2);
    } 
    else {
            decoderlogger.log(LogLevel::debug) << "AVDecoder: registered decoder with av_codec_id "<<av_codec_id<<std::endl;
    }
}


AVDecoder::~AVDecoder() {
  // https://stackoverflow.com/questions/14914462/ffmpeg-memory-leak
  av_free_packet(av_packet);
  
  // av_frame_free(&av_frame);
  // av_free(av_frame); // needs this as well?
  
  avcodec_close(av_codec_context);
  avcodec_free_context(&av_codec_context);
  
  delete av_packet;
}


void AVDecoder::flush() {
  avcodec_flush_buffers(av_codec_context);
}



VideoDecoder::VideoDecoder(AVCodecID av_codec_id, int n_threads) : AVDecoder(av_codec_id, n_threads), width(0), height(0) { // , av_pixel_format(AV_PIX_FMT_NONE) {
  // out_frame is AVBitmapFrame
  /*
  // decoder slow down simulation
  gen =std::mt19937(rd());
  dis =std::uniform_int_distribution<>(0,5);
  */
};


VideoDecoder::~VideoDecoder() {
};


Frame* VideoDecoder::output() {
  return &out_frame;
};


bool VideoDecoder::pull() {
  int retcode;
  int got_frame;
  
  has_frame = false;
  
  /* // some debugging .. (filter just sps, pps and keyframes)
  if (in_frame.h264_pars.slice_type != H264SliceType::pps and in_frame.h264_pars.slice_type != H264SliceType::sps and in_frame.h264_pars.slice_type != H264SliceType::i) {
    return false;
  }
  */
  
  av_packet->data =in_frame.payload.data(); // pointer to payload
  av_packet->size =in_frame.payload.size();
  
#ifdef DECODE_VERBOSE
  std::cout << "VideoDecoder: pull: size    ="  <<av_packet->size<<std::endl;
  std::cout << "VideoDecoder: pull: payload =[" << in_frame.dumpPayload() << "]" << std::endl;
#endif
  
#ifdef DECODE_TIMING
  long int mstime=getCurrentMsTimestamp();
#endif
  
  
/*
#ifdef AV_REALLOC
  av_frame_free(&av_frame);
  av_frame =av_frame_alloc();
#endif
*/
  
  retcode = avcodec_decode_video2(av_codec_context, out_frame.av_frame, &got_frame, av_packet);
  
  if (retcode < 0) {
    decoderlogger.log(LogLevel::crazy) << "VideoDecoder: decoder error " << retcode << std::endl;
    return false;
  }
  
  /*
  retcode = avcodec_send_packet(av_codec_context, av_packet); // new API
  avcodec_receive_frame(av_codec_context, out_frame.av_frame); // new API
  */
  
  AVFrame       *av_frame         =out_frame.av_frame;
  AVPixelFormat &new_pixel_format =av_codec_context->pix_fmt;
  
  if (new_pixel_format==AV_PIX_FMT_YUV420P) { // ok
  }
  else if (new_pixel_format==AV_PIX_FMT_YUVJ420P) { // ok
  }
  else { // TODO: set up scaling context for converting other YUV formats to YUV420P
#ifdef DECODE_VERBOSE
    std::cout << "VideoDecoder: pull: wrong pixel format " << int(new_pixel_format) << std::endl;
#endif    
    return false;
  }
  
  if (av_frame->width<1 or av_frame->height<1) { // crap
#ifdef DECODE_VERBOSE
    std::cout << "VideoDecoder: pull: corrupt frame: w, h = " <<  av_frame->width << " " << av_frame->height << std::endl;
#endif
    return false;
  }
  
  if ( height!=av_frame->height || width!=av_frame->width || out_frame.av_pixel_format!=new_pixel_format ) { // update aux objects!
    out_frame.av_pixel_format=new_pixel_format;
    out_frame.update(); // uses av_frame and out_frame.av_pixel_format
  }
  out_frame.copyMetaFrom(&in_frame);  // after this, the AVBitmapFrame instance is ready to go..
  // out_frame.mstimestamp =av_frame->pts; // timestamp is now the presentation timestamp given by the decoder .. uh-oh need to convert from time_base units of AVStream
  
/*
#ifdef AV_REALLOC
  av_free_packet(av_packet);
  av_init_packet(av_packet);
#endif
*/
  
#ifdef DECODE_TIMING
  mstime=getCurrentMsTimestamp()-mstime;
  if (mstime>40) {
    std::cout << "VideoDecoder: pull: decoding took " << mstime << "milliseconds" << std::endl;
  }
#endif
  
  /*
  // debugging: let's simulate slow decoding
  int delay=dis(gen)+40;
  std::cout << "VideoDecoder: pull: delay=" << delay << std::endl;
  std::this_thread::sleep_for(std::chrono::milliseconds(delay));
  */
  
  
#ifdef DECODE_VERBOSE
  std::cout << "VideoDecoder: pull: retcode, got_frame, AVBitmapFrame : " << retcode << " " << got_frame << " " << out_frame << std::endl;
#endif
  
#ifdef DECODE_VERBOSE
  std::cout << "VideoDecoder: decoded: " << retcode <<" bytes => "<< out_frame << std::endl;
  std::cout << "VideoDecoder: payload: "<< out_frame.dumpPayload() << std::endl;
#endif
  
  has_frame = true;
  return true;
}

