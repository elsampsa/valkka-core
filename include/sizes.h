#ifndef SIZES_HEADER_GUARD 
#define SIZES_HEADER_GUARD

/*
 * sizes.h : Default payload sizes and other numerical constants
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
 *  @file    sizes.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.0 
 *  
 *  @brief Default payload sizes and other numerical constants
 *
 */

#include "common.h"

static const int VERSION_MAJOR = 0; // <pyapi>
static const int VERSION_MINOR = 3; // <pyapi>
static const int VERSION_PATCH = 0; // <pyapi>


namespace Timeouts { ///< Various thread timeouts in milliseconds
  const static long unsigned thread       =250; // Timeouts::thread
  const static long unsigned livethread   =250; // Timeouts::livethread
  const static long unsigned avthread     =250; // Timeouts::avthread
  const static long unsigned openglthread =250; // Timeouts::openglthread
  const static long int filethread        =2000; // Timeouts::filethread
}


enum PayloadSizes {
  DEFAULT_PAYLOAD_SIZE            = 1024,    ///< Default buffer size in Live555 for h264
  // DEFAULT_PAYLOAD_SIZE_H264       = 1024*100, ///< Default buffer size in Live555 for h264
  DEFAULT_PAYLOAD_SIZE_H264       = 1024*300, ///< Default buffer size in Live555 for h264 // debug
  // DEFAULT_PAYLOAD_SIZE_H264       = 1024*10,  ///< Default buffer size in Live555 for h264 // debug
  DEFAULT_PAYLOAD_SIZE_PCMU       = 1024,    ///< Default buffer size in Live555 for pcmu
  DEFAULT_FRAME_FIFO_PAYLOAD_SIZE = 1024*30  ///< Fifo buffers are this size by default
  // DEFAULT_FRAME_FIFO_PAYLOAD_SIZE = 1024*300  ///< Fifo buffers are this size by default // debug
  // DEFAULT_PAYLOAD_SIZE_H264       = 1024*10,  ///< Fifo buffers are this size by default // debug
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

static const std::vector<uint8_t> nalstamp = {0,0,0,1};


#endif