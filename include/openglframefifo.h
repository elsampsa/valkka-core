#ifndef openglframefifo_HEADER_GUARD
#define openglframefifo_HEADER_GUARD
/*
 * openglframefifo.h : FrameFifo for OpenGLThread: stack of YUV frames and uploading to GPU
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
 *  @file    openglthread.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.10.0 
 *  
 *  @brief FrameFifo for OpenGLThread: stack of YUV frames and uploading to GPU
 *
 */ 

#include "opengl.h"
#include "framefilter.h"

/** Describes the stack structure and fifo behaviour for an OpenGLFrameFifo
 * 
 * @ingroup queues_tag
 * @ingroup openglthread_tag
 */
struct OpenGLFrameFifoContext {                                                                                                     // <pyapi>
  OpenGLFrameFifoContext() : n_720p(20), n_1080p(20), n_1440p(20), n_4K(20), n_setup(20), n_signal(20), flush_when_full(DEFAULT_OPENGLFRAMEFIFO_FLUSH_WHEN_FULL) {};  // <pyapi>
  int n_720p;                                                                                                                       // <pyapi>
  int n_1080p;                                                                                                                      // <pyapi>
  int n_1440p;                                                                                                                      // <pyapi>
  int n_4K;                                                                                                                         // <pyapi>
  int n_setup;     ///< setup data                                                                                                  // <pyapi>
  int n_signal;    ///< signals OpenGLThread                                                                                        // <pyapi>
  bool flush_when_full; ///< Flush when filled                                                                                      // <pyapi>
};                                                                                                                                  // <pyapi>


std::ostream& operator<< (std::ostream& os, const OpenGLFrameFifoContext& ctx) {
  return os << "720p: " << ctx.n_720p <<" / 1080p: " << ctx.n_1080p << " / 1440p: " << ctx.n_1440p << " / 4K: " << ctx.n_4K << " / n_setup: " << ctx.n_setup << " / n_signal: " << ctx.n_signal << " / flush_when_full: " << int(ctx.flush_when_full);
}


/** A FrameFifo managed and used by OpenGLThread.
 * 
 * Manages a stack of (pre-reserved) YUVFrame instances for different resolutions.
 *  
 * @ingroup openglthread_tag
 * @ingroup queues_tag
 */
class OpenGLFrameFifo : public FrameFifo { // <pyapi>
  
friend class OpenGLThread; // can manipulate reservoirs, stacks, etc.
  
public: // <pyapi>
  /** Default constructor
   * 
   */
  OpenGLFrameFifo(OpenGLFrameFifoContext ctx=OpenGLFrameFifoContext()); // <pyapi>
  /** Default destructor
   */
  ~OpenGLFrameFifo(); // <pyapi>
  
protected:
  OpenGLFrameFifoContext             gl_ctx;          ///< Stack profile and overflow behaviour
  std::map<BitmapType,YUVReservoir>  yuv_reservoirs;  ///< Instances of YUVFrame s
  std::map<BitmapType,YUVStack>      yuv_stacks;      ///< Pointers to Frames s in the reservoirs
  // note: we still have "reservoirs" and "stacks" inherited from FrameFifo

protected:
  YUVFrame* prepareAVBitmapFrame(AVBitmapFrame* frame);    ///< Tries to get a YUVFrame with correct bitmap dimensions from the stack.  If success, returns pointer to the YUVFrame, otherwise NULL
  
public: // redefined virtual
  virtual bool writeCopy(Frame* f, bool wait=false);     ///< Redefined. Uses FrameFifo::writeCopy.  Separates configuration frames and YUVFrames.
  virtual void recycle_(Frame* f);                       ///< Redefined. Uses FrameFifo::recycle_. Separates configuration frames and YUVFrames.
  
public:
  void allocateYUV();     ///< Allocate YUVFrame's .. must be done after OpenGLThread has been started
  void deallocateYUV();   ///< Deallocate YUVFrame's .. must be done before OpenGLThread exits
  void dumpYUVStacks();    ///< Dump frames in OpenGLFrameFifo::yuv_stacks
  void YUVdiagnosis();     ///< Brief resumen of OpenGLFrameFifo::yuv_stacks
  
public: // setters
  void debugOn() {debug=true;}
  void debugOff(){debug=false;}
  
private:
  bool debug;
}; // <pyapi>

#endif