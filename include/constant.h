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
 *  @version 0.11.0 
 *  
 *  @brief   Constant/default values, version numbers
 */ 


#include "common.h"

#define GLX_CONTEXT_MAJOR_VERSION       0x2091
#define GLX_CONTEXT_MINOR_VERSION       0x2092

#define ALIGNMENT 1
/*
 
https://stackoverflow.com/questions/35678041/what-is-linesize-alignment-meaning
https://softwareengineering.stackexchange.com/questions/328775/how-important-is-memory-alignment-does-it-still-matter
 
- decoder returns YUV frames that are aligned
- ..during YUV => RGB interpolation (SwScaleFrameFilter, etc.), we get rid of the alignment (i.e. the extra padding bytes)
- ..anyway, that must be done at some moment before passing the frames downstream (for analyzers, etc.)
 
=> KEEP ALIGNMENT = 1

- ..there might be performance benefits in using, for the final rgb bitmap images, widths that are multiples of 32
*/

const char* get_numpy_version() {  // <pyapi>
    return NUMPY_VERSION;          // <pyapi>
}                                  // <pyapi> 
    
static const int VERSION_MAJOR = 0; // <pyapi>
static const int VERSION_MINOR = 11; // <pyapi>
static const int VERSION_PATCH = 0; // <pyapi>

static const unsigned DEFAULT_OPENGLTHREAD_BUFFERING_TIME  = 300;   // in milliseconds // <pyapi>
static const bool DEFAULT_FRAMEFIFO_FLUSH_WHEN_FULL        = false; // <pyapi>
static const bool DEFAULT_OPENGLFRAMEFIFO_FLUSH_WHEN_FULL  = false; // <pyapi>
static const long int DEFAULT_TIMESTAMP_RESET_TIME         = 60000; // <pyapi>
static const long int TIMESTAMP_CORRECT_TRESHOLD           = 30000; // <pyapi> // timestamp correctors start correcting timestamps if they are this much off (in milliseconds)
static const std::size_t FS_GRAIN_SIZE                     = 4096;  // <pyapi> // grain size for ValkkaFS

namespace Timeout { ///< Various thread timeouts in milliseconds
  const static long unsigned thread       =250; // Timeout::thread
  const static long unsigned livethread   =250; // Timeout::livethread
  const static long unsigned avthread     =250; // Timeout::avthread
  const static long unsigned openglthread =250; // Timeout::openglthread
  const static long unsigned valkkafswriterthread = 250; // Timeout::valkkafswriterthread
  const static long unsigned valkkafsreaderthread = 250; // Timeout::valkkafswriterthread
  const static long unsigned filecachethread = 1000; // Timeout::valkkacachethread
  // const static long unsigned filecachethread = 500; // Timeout::valkkacachethread
  const static long unsigned usbthread    =250; // Timeout::usbthread
  const static long int filethread        =2000; // Timeout::filethread
}


enum PayloadSizes {
  DEFAULT_PAYLOAD_SIZE            = 1024,    ///< Default buffer size in Live555 for h264 // debug
  
  // DEFAULT_PAYLOAD_SIZE_H264       = 1024*100, ///< Default buffer size in Live555 for h264
  DEFAULT_PAYLOAD_SIZE_H264       = 1024*300, ///< Default buffer size in Live555 for h264 
  // DEFAULT_PAYLOAD_SIZE_H264       = 1024*500, ///< Default buffer size in Live555 for h264
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

typedef unsigned       BitmapType; 
typedef unsigned short SlotNumber;   // <pyapi>
typedef std::size_t    IdNumber;     // <pyapi>

static const SlotNumber I_MAX_SLOTS = 255; // Slot number maximum index.  Max number of slots = I_MAX_SLOTS+1
static const int I_MAX_SUBSESSIONS = 3;


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

bool operator==(BitmapPars const &a, BitmapPars const &b) { // is copyable ?
    return ( (a.y_linesize == b.y_linesize) and (a.u_linesize == b.u_linesize) and (a.v_linesize == b.v_linesize) );
}

inline std::ostream &operator<<(std::ostream &os, BitmapPars const &m) {
  return os << "<BitmapPars: type=" << int(m.type) << " w, h=" << m.width << ", " << m.height << ">";
}

static const BitmapPars N720  (1,1280,720, 2,2);
static const BitmapPars N1080 (2,1920,1080,2,2);

// static const BitmapPars N1440 (3,2560,1440,2,2); // this is 2K, but not "universal" one, where all 2Ks would fit
static const BitmapPars N1440 (3,3000,1690,2,2); // all 2Ks will fit here  :)   Takes more than two times less space than 4K

static const BitmapPars N4K   (4,4032,3000,2,2);

static const std::vector<uint8_t> nalstamp = {0,0,0,1};


#endif
