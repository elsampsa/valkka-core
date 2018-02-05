/*
 * decoders.h : FFmpeg decoders
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
 *  @file    decoders.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.0 
 *  
 *  @brief FFmpeg decoders
 * 
 */ 

#include "frames.h"
#include <random>


/** A Virtual class for decoders
 * 
 * Inspects Frame in_frame, does something, and writes to Frame out_frame
 * 
 * 
 * @ingroup decoding_tag
 */
class DecoderBase {
  
public:
  DecoderBase();          ///< Default constructor
  virtual ~DecoderBase(); ///< Default destructor
    
public:
  Frame  in_frame;  ///< Encoded frame payload, etc. put here
  Frame  out_frame; ///< Decoded frame appears here
  
public:
  virtual void flush() =0; ///< Reset decoder state
  virtual bool pull()  =0; ///< Decode in_frame to out_frame.  Return true if decoder returned a new frame (into out_frame), otherwise false
};


/** A Dummy decoder
 * 
 * Simply copies in_frame to out_frame
 * 
 * @ingroup decoding_tag
 */
class DummyDecoder : public DecoderBase {

  
public:
  void flush();
  bool pull();
};


/** Decoder using FFmpeg/libav
 * 
 * A Generic class to decode any media format with FFmpeg
 * 
 * @ingroup decoding_tag
 */
class Decoder : public DecoderBase {
  
public:
  /** Default constructor
   * 
   * @param av_codec_id  FFmpeg AVCodecId identifying the codec
   * 
   */
  Decoder(AVCodecID av_codec_id);
  virtual ~Decoder();
    
public:
  AVCodecID       av_codec_id;       ///< FFmpeg AVCodecId, identifying the codec 
  AVPacket*       av_packet;         ///< FFmpeg internal data structure; encoded frame (say, H264)
  AVFrame*        av_frame;          ///< FFmpeg internal data structure: decoded frame (say, YUV bitmap)
  AVCodec*        av_codec;          ///< FFmpeg internal data structure
  AVCodecContext* av_codec_context;  ///< FFmpeg internal data structure
  
public:
  void flush();  
};


/** Video decoder using FFmpeg/libav
 * 
 * Decode video with FFmpeg, place decoded data to out_frame.  VideoDecoder.out_frame's Frame.frametype is set to FrameType::avframe, i.e. the consumer of this frame must immediately copy the data.
 * 
 * See also \ref pipeline
 * 
 * @ingroup decoding_tag
 */
class VideoDecoder : public Decoder {
  
public:
  VideoDecoder(AVCodecID av_codec_id); ///< @copydoc Decoder::Decoder
  virtual ~VideoDecoder(); ///< Default destructor
      
public:
  bool pull(); ///< @copydoc Decoder::pull
  
/*
private: // for simulating decoder slow down
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937       gen; //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dis;
*/
  
};






