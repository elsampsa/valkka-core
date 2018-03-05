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
 *  @version 0.3.0 
 *  @brief   FFmpeg decoders
 */ 

#include "decoders.h"
#include "logging.h"

// https://stackoverflow.com/questions/14914462/ffmpeg-memory-leak
#define AV_REALLOC

DecoderBase::DecoderBase() {};
DecoderBase::~DecoderBase() {};


void DummyDecoder::flush() {
};


bool DummyDecoder::pull() {
  out_frame=in_frame;
  return true;
}



Decoder::Decoder(AVCodecID av_codec_id) : av_codec_id(av_codec_id) {
  int retcode;

  // av_packet       =av_packet_alloc(); // oops.. not part of the public API (defined in avcodec.c instead of avcodec.h)
  av_packet       =new AVPacket();
  av_init_packet(av_packet);
  av_frame        =av_frame_alloc();
  
  av_codec        =avcodec_find_decoder(av_codec_id);
  if (!av_codec) {
    std::perror("Decoder: FATAL: could not find av_codec");
    exit(2);
  }
  av_codec_context=avcodec_alloc_context3(av_codec);
  retcode         =avcodec_open2(av_codec_context, av_codec, NULL); 
  // is this going to change at some moment .. ? https://blogs.gentoo.org/lu_zero/2016/03/29/new-avcodec-api/
  // aha..! it's here: https://ffmpeg.org/doxygen/3.2/group__lavc__decoding.html#ga58bc4bf1e0ac59e27362597e467efff3
  if (retcode!=0) {
    std::perror("Decoder: FATAL could not open codec");
    exit(2);
  } 
  else {
    decoderlogger.log(LogLevel::debug) << "Decoder: registered decoder with av_codec_id "<<av_codec_id<<std::endl;
  }
  // av_packet->data =(uint8_t*)(in_frame.payload.data()); // pointer to payload
}


Decoder::~Decoder() {
  // https://stackoverflow.com/questions/14914462/ffmpeg-memory-leak
  av_free_packet(av_packet);
  
  av_frame_free(&av_frame);
  av_free(av_frame); // needs this as well?
  
  avcodec_close(av_codec_context);
  avcodec_free_context(&av_codec_context);
  
  delete av_packet;
}


void Decoder::flush() {
  avcodec_flush_buffers(av_codec_context);
}



VideoDecoder::VideoDecoder(AVCodecID av_codec_id) : Decoder(av_codec_id) {
  /*
  // decoder slow down simulation
  gen =std::mt19937(rd());
  dis =std::uniform_int_distribution<>(0,5);
  */
  
};
VideoDecoder::~VideoDecoder() {
};
  

bool VideoDecoder::pull() {
  int retcode;
  int got_frame;
  
  /* // some debugging .. (filter just sps, pps and keyframes)
  if (in_frame.h264_pars.slice_type != H264SliceType::pps and in_frame.h264_pars.slice_type != H264SliceType::sps and in_frame.h264_pars.slice_type != H264SliceType::i) {
    return false;
  }
  */
  
  av_packet->data =in_frame.payload.data(); // pointer to payload
  av_packet->size =in_frame.payload.size();
  
#ifdef DECODE_VERBOSE
  std::cout << "VideoDecoder: pull: size    ="<<av_packet->size<<std::endl;
  std::cout << "VideoDecoder: pull: payload =[" << in_frame.dumpPayload() << "]" << std::endl;
#endif
  
#ifdef DECODE_TIMING
  long int mstime=getCurrentMsTimestamp();
#endif
  
  
#ifdef AV_REALLOC
  av_frame_free(&av_frame);
  av_frame =av_frame_alloc();
#endif
  
  retcode=avcodec_decode_video2(av_codec_context,av_frame,&got_frame,av_packet);

#ifdef AV_REALLOC
  av_free_packet(av_packet);
  av_init_packet(av_packet);
#endif
  
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
  std::cout << "VideoDecoder: pull: retcode, got_frame, av_frame->width : " << retcode << " " << got_frame << " " << av_frame->width << std::endl;
#endif
  
  if (retcode<0) {
    decoderlogger.log(LogLevel::crazy) << "VideoDecoder: decoder error " << retcode << std::endl;
    return false;
  }
  /*
  else if (retcode==0) {
#ifdef DECODE_VERBOSE
    std::cout << "VideoDecoder: no frame" << std::endl;
#endif
    return false;
  }
  */
  if (av_frame->width==0) {
    decoderlogger.log(LogLevel::debug) << "VideoDecoder: corrupt frame" << std::endl;
    return false;
  }
  
  out_frame.reset();
  in_frame.copyMeta(&out_frame);
  out_frame.av_codec_context =av_codec_context; // apart from this point, it is up to the filterchain to handle the ffmpeg AVFrame structure and to use the info in av_codec_context
  out_frame.av_frame         =av_frame;                // av_frame and av_codec_context are reserved and re-used by the Decoder instance
  out_frame.frametype        =FrameType::avframe;
#ifdef DECODE_VERBOSE
  std::cout << "VideoDecoder: decoded: " << retcode <<" bytes => "<< out_frame << std::endl;
  // av_frame and av_codec_context are reserved and re-used by the Decoder instances.  Must copy them!  .. that is done finally, by the OpenGLFrameFifo::writeCopy
  std::cout << "VideoDecoder: payload: "<< int(av_frame->data[0][0]) << " " << int(av_frame->data[1][0]) << " " << int(av_frame->data[2][0]) << std::endl;
#endif
  return true;
}
