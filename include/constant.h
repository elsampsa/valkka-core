#ifndef constant_HEADER_GUARD
#define constant_HEADER_GUARD
/*
 * constant.h : Constant/default values, version numbers
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
 *  @file    constant.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.4.5 
 *  
 *  @brief   Constant/default values, version numbers
 */ 


#include "common.h"

static const int VERSION_MAJOR = 0; // <pyapi>
static const int VERSION_MINOR = 4; // <pyapi>
static const int VERSION_PATCH = 5; // <pyapi>

static const unsigned DEFAULT_OPENGLTHREAD_BUFFERING_TIME  = 300;   // in milliseconds // <pyapi>
static const bool DEFAULT_FRAMEFIFO_FLUSH_WHEN_FULL        = false; // <pyapi>
static const bool DEFAULT_OPENGLFRAMEFIFO_FLUSH_WHEN_FULL  = false; // <pyapi>
static const long int DEFAULT_TIMESTAMP_RESET_TIME         = 60000; // <pyapi>

namespace Timeout { ///< Various thread timeouts in milliseconds
  const static long unsigned thread       =250; // Timeout::thread
  const static long unsigned livethread   =250; // Timeout::livethread
  const static long unsigned avthread     =250; // Timeout::avthread
  const static long unsigned openglthread =250; // Timeout::openglthread
  const static long int filethread        =2000; // Timeout::filethread
}


enum PayloadSizes {
  DEFAULT_PAYLOAD_SIZE            = 1024,    ///< Default buffer size in Live555 for h264 // debug
  
  // DEFAULT_PAYLOAD_SIZE_H264       = 1024*100, ///< Default buffer size in Live555 for h264
  DEFAULT_PAYLOAD_SIZE_H264       = 1024*300, ///< Default buffer size in Live555 for h264 
  // DEFAULT_PAYLOAD_SIZE_H264       = 1024*10,  ///< Default buffer size in Live555 for h264 // debug
  // DEFAULT_PAYLOAD_SIZE_H264       = 1024, // use this small value for debugging (( debug
  
  DEFAULT_PAYLOAD_SIZE_PCMU       = 1024,    ///< Default buffer size in Live555 for pcmu
};

enum MaxSizes {
  N_MAX_SLOTS           =256,  ///< Maximum number of slots (used both by livethread.cpp and openglthread.cpp
  N_MAX_DECODERS        =4,    ///< Maximum number of decoders per one AVThread instance
  // N_MAX_GPU_STACK       =200 // not used..
};


// using BitmapType = unsigned;        
// using SlotNumber = unsigned short; // swig does not get this ..

typedef unsigned BitmapType; 
typedef unsigned short SlotNumber;   // <pyapi>

static const SlotNumber I_MAX_SLOTS = 255; // Slot number maximum index.  Max number of slots = I_MAX_SLOTS+1


/** 
 * For AVBitmapFrames, linesizes are the widths + padding bytes
 * 
 * For pre-reserved YUVFrames width and linesize are always the same.
 * 
 * Texture dimensions match always YUVFrame dimensions
 * 
 */
struct BitmapPars {
  BitmapPars(BitmapType type=0, int width=0, int height=0, int w_fac=1, int h_fac=1) : type(type), width(width), height(height), w_fac(w_fac), h_fac(h_fac) {
    y_size =width*height;
    u_size =y_size/w_fac/h_fac;
    v_size =u_size;
    
    // linesize aka maximum allowed width
    y_width  =width;
    u_width  =width/w_fac;
    v_width  =u_width;
    
    y_height  =height;
    u_height  =height/h_fac;
    v_height  =u_height;
    
    // the default
    y_linesize =y_width;
    u_linesize =u_width;
    v_linesize =v_width;
  };
  
  BitmapType    type;  
  
  int           width;  ///< width of luma plane
  int           height; ///< height of luma plane
  int           w_fac;  ///< width factor for chroma plane
  int           h_fac;  ///< height factor for chroma plane
  
  int           y_size;
  int           u_size;
  int           v_size;
  
  int           y_width;
  int           u_width;
  int           v_width;
  
  int           y_height;
  int           u_height;
  int           v_height;
  
  int           y_linesize;
  int           u_linesize;
  int           v_linesize;
};

std::ostream &operator<<(std::ostream &os, BitmapPars const &m) {
  return os << "<BitmapPars: type=" << int(m.type) << " w, h=" << m.width << ", " << m.height << ">";
}

static const BitmapPars N720  (1,1280,720, 2,2);
static const BitmapPars N1080 (2,1920,1080,2,2);
static const BitmapPars N1440 (3,2560,1440,2,2);
static const BitmapPars N4K   (4,4032,3000,2,2);





/*
namespace BitmapPars {
  const static BitmapType notype     =0;
  
  namespace N720 {
    const static BitmapType type     =1; // like enum class
    const static unsigned w          =1280;
    const static unsigned h          =720;
    const static unsigned size       =1280*720;
    const static unsigned yuvsize    =1280*720*3/2;
 };
 namespace N1080 {
    const static BitmapType type     =2; 
    const static unsigned w          =1920;
    const static unsigned h          =1080;
    const static unsigned size       =1920*1080; 
    // 1280*960 uses this .. takes a frame from stack with YUVPBO .. y_size, u_size, v_size = 1920*1080, etc.
    const static unsigned yuvsize    =1920*1080*3/2;
 };
 namespace N1440 {
    const static BitmapType type     =3; 
    const static unsigned w          =2560;
    const static unsigned h          =1440;
    const static unsigned size       =2560*1440;
    const static unsigned yuvsize    =2560*1440*3/2;
 };
 namespace N4K {
    const static BitmapType type     =4; 
    const static unsigned w          =4032;
    const static unsigned h          =3000;
    const static unsigned size       =4032*3000;
    const static unsigned yuvsize    =4032*3000*3/2;
 }; 
};
*/

static const std::vector<uint8_t> nalstamp = {0,0,0,1};


#endif
