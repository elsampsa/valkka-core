#ifndef rgbframefifo_HEADER_GUARD
#define rgbframefifo_HEADER_GUARD
/*
 * rgbframefifo.h :
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
 *  @file    rgbframefifo.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.2.2 
 *  
 *  @brief
 */ 

#include "framefifo.h"



struct RGBFrameFifoContext {
    RGBFrameFifoContext(int max_width, int max_height, int n) : max_width(max_width), max_height(max_height), n(n), flush_when_full(false) {};
    int max_width;
    int max_height;
    int n;
    bool flush_when_full;
};


/** A FrameFifo for RGBFrame s 
 * 
 * - Manages a stack of (pre-reserved) RGBFrame s
 * - Only frames with a defined maximum width & height are accepted
 *  
 * @ingroup queues_tag
 */
class RGBFrameFifo : public FrameFifo {
    
public:
    /** Default constructor
    * 
    */
    // RGBFrameFifo(int max_width, int max_height, int n); // <pyapi>
    RGBFrameFifo(RGBFrameFifoContext ctx); // <pyapi>
    /** Default destructor
    */
    ~RGBFrameFifo(); // <pyapi>
    
protected:
    RGBReservoir    rgb_reservoir;
    RGBStack        rgb_stack;
    // note: we still have "reservoirs" and "stacks" inherited from FrameFifo

public: // redefined virtual
    virtual bool writeCopy(Frame* f, bool wait=false);     ///< Redefined. Uses FrameFifo::writeCopy.  Separates configuration frames and YUVFrames.
    virtual void recycle_(Frame* f);                       ///< Redefined. Uses FrameFifo::recycle_. Separates configuration frames and YUVFrames.
    
}; // <pyapi>



#endif
