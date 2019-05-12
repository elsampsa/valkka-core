#ifndef movement_HEADER_GUARD
#define movement_HEADER_GUARD
/*
 * movement.h : Framefilter implementing a movement detector
 * 
 * Copyright 2017-2019 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * Authors: Petri Eranko <petri.eranko@dasys.fi>
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
 *  @file    movement.h
 *  @author  Petri Eranko
 *  @date    2019
 *  @version 0.11.0 
 *  
 *  @brief   Framefilter implementing a movement detector
 */ 


#include "framefilter.h"
#include "Python.h"

class MovementFrameFilter : public FrameFilter {                                           // <pyapi>
  
public:                                                                                    // <pyapi>
    /**
     * @param interval  How often the frame is checked (milliseconds)
     * @param treshold  Treshold value for movement
     * @param duration  Duration of a single event (milliseconds)
     * 
     */
    MovementFrameFilter(const char* name, long int interval, float treshold, long int duration, PyObject* pycallback, FrameFilter* next=NULL);     // <pyapi>
    virtual ~MovementFrameFilter(); // <pyapi>
    
protected:
    long int  movement_start; ///< State of the movement detector & the timestamp when movement was first detected
    long int  mstimestamp;    ///< Saved timestamp: time of the previous frame
    long int  duration;       ///< Movement events within this time are the same event
    long int  interval;       ///< How often check the frames
    int       y_size;         ///< Size of the last cached luma plane
    uint8_t*  luma;           ///< Cached luma component of the yuv image
    PyObject* pycallback;    ///< Python method to be called at movement & still events
    float     treshold;
  
  
protected:
    /** Takes AVBitmapFrame */
    void run(Frame* frame);
    void go(Frame* frame);
    
public:
    void reset();                                                                         // <pyapi>
};                                                                                        // <pyapi>



#endif
