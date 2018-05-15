#ifndef decoder_HEADER_GUARD 
#define decoder_HEADER_GUARD
/*
 * decoder.h : FFmpeg decoders
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
 *  @file    decoder.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.4.3 
 *  
 *  @brief FFmpeg decoders
 * 
 */ 

#include "frame.h"
#include <random>


// AVThread has std::vector<Decoder*> decoders
// decoders[1]->output() returns reference to Decoder::out_frame
// (VideoDecoder*)(..) returns reference to VideoDecoder::out_frame, right? (Decoder::out_frame) is hidden
// 
// decoders[1] .. is a VideoDecoder.. but decoders has been declared as a vector Decoder
// Decoder* decoder = VideoDecoder()
// .. so, decoder->getWhatever, where getWhatever is virtual method, will always give the correct object
// but decoder->out_frame returns frame that depends on the cast




// VideoDecoder->output() returns 


/** A Virtual class for decoders
 * 
 * Inspects Frame in_frame, does something, and writes to Frame out_frame
 * 
 * 
 * @ingroup decoding_tag
 */
class Decoder {
  
public:
  Decoder();          ///< Default constructor
  virtual ~Decoder(); ///< Default destructor
    
protected:
  BasicFrame in_frame; ///< Payload data to be decoded.
  
public:
  void input(Frame *f); ///< Create a copy of the frame into the internal storage of the decoder (i.e. to Decoder::in_frame)
  long int getMsTimestamp(); /// < Return in_frame timestamp
  virtual Frame* output() =0; ///< Return a reference to the internal storage of the decoder where the decoded frame is.  The exact frametype depends on the Decoder class (and decoder library)
  virtual void flush() =0; ///< Reset decoder state.  How to flush depends on the decoder library
  virtual bool pull()  =0; ///< Decode in_frame to out_frame.  Return true if decoder returned a new frame (into out_frame), otherwise false.  Implementation depends on the decoder library.
};


/** A Dummy decoder
 * 
 * Implements a "dummy" decoding library: simply copies in_frame to out_frame.  DummyDecoder::out_frame is a BasicFrame.
 * 
 * @ingroup decoding_tag
 */
class DummyDecoder : public Decoder {

private:
  BasicFrame    out_frame; ///< Output frame: no decoding, just copy input here
  
public:
  virtual Frame* output();  ///< Return a reference to the internal storage of the decoder where the decoded frame is
  virtual void flush();
  virtual bool pull();
};


/** Decoder using FFmpeg/libav
 * 
 * A virtual class to decode any media format with FFmpeg
 * 
 * @ingroup decoding_tag
 */
class AVDecoder : public Decoder {
  
public:
  /** Default constructor
   * 
   * @param av_codec_id  FFmpeg AVCodecId identifying the codec
   * 
   */
  AVDecoder(AVCodecID av_codec_id);
  virtual ~AVDecoder();
    
  
public:
  AVCodecID       av_codec_id;       ///< FFmpeg AVCodecId, identifying the codec 
  AVPacket*       av_packet;         ///< FFmpeg internal data structure; encoded frame (say, H264)
  AVCodecContext* av_codec_context;  ///< FFmpeg internal data structure
  AVCodec*        av_codec;          ///< FFmpeg internal data structure
  
public:
  virtual void flush();  
};


/** Video decoder using FFmpeg/libav
 * 
 * Decode video with FFmpeg, place decoded data to out_frame.  VideoDecoder.out_frame's Frame.frametype is set to FrameType::avframe, i.e. the consumer of this frame must immediately copy the data.
 * 
 * See also \ref pipeline
 * 
 * @ingroup decoding_tag
 */
class VideoDecoder : public AVDecoder {
  
public:
  VideoDecoder(AVCodecID av_codec_id); ///< Default constructor
  virtual ~VideoDecoder();             ///< Default destructor
  
protected:
  AVBitmapFrame out_frame;
  int width;
  int height;
  // AVPixelFormat av_pixel_format;
  
public:
  virtual Frame* output();      ///< Return a reference to the internal storage of the decoder where the decoded frame is
  virtual bool pull();
  
/*
private: // for simulating decoder slow down
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937       gen; //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dis;
*/
  
};

#endif
