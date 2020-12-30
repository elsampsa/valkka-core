#ifndef codec_HEADER_GUARD 
#define codec_HEADER_GUARD
/*
 * codec.h : Codec definitions (slightly outdated)
 * 
 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>
 *
 */

/** 
 *  @file    codec.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.1.0 
 *  
 *  @brief   Codec definitions (slightly outdated)
 */ 


/** Outdated.  We're using FFmpeg/libav parameters instead
 * 
 */
enum class MediaType {
  none,
  video,
  audio
};

/** Outdated.  We're using FFmpeg/libav parameters instead
 * 
 */
enum class Codec {
  none,
  h264,
  yuv,
  rgb,
  pcmu
};


/** Various H264 frame types
 * 
 * @ingroup frames_tag
 */
namespace H264SliceType {
  static const unsigned none  =0;
  static const unsigned sps   =7;
  static const unsigned pps   =8;
  static const unsigned i     =5;
  static const unsigned pb    =1;
};


struct H264Pars { ///< H264 parameters
  H264Pars() : slice_type(H264SliceType::none) {};
  short unsigned slice_type;
};
inline std::ostream &operator<<(std::ostream &os, H264Pars const &m) {
 return os << "H264: slice_type="<<m.slice_type;
}


struct SetupPars { ///< Setup parameters for decoders and muxers (outdated)
  // AVCodecID codec_id; //https://ffmpeg.org/doxygen/3.0/group__lavc__core.html#gaadca229ad2c20e060a14fec08a5cc7ce
  SetupPars() : mediatype(MediaType::none), codec(Codec::none) {};
  MediaType mediatype;
  Codec     codec;
};
inline std::ostream &operator<<(std::ostream &os, SetupPars const &m) {
 return os << "Setup: mediatype="<< int(m.mediatype) << " codec="<< int(m.codec);
}

#endif
