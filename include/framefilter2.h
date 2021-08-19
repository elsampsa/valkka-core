#ifndef framefilter2_HEADER_GUARD
#define framefilter2_HEADER_GUARD
/*
 * framefilter2.h : More framefilters
 * 
 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
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
 *  @file    framefilter2.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.2.2 
 *  
 *  @brief  More framfilters
 */ 

#include "framefilter.h"
#include "Python.h"


/** LiveThread sends a special frame if a camera is detected offline.
 * 
 * This framefilter catches that frame and executes a python callback
 * 
 * Your callback should be a python function like this:
 *
 \verbatim 
 def cb(tup):
    slot = tup[0]
    # congrats: now you have the slot that has some 
    # problems with the rtsp stream
    print("problems at slot", slot)
    # this python callback is launched from the cpp side
    # so it is a good idea to exit asap so that your callback chain
    # from cpp side exists asap
 \endverbatim
 * 
 * Remember also that the frequency of the check is set by LiveConnectionContext.mstimeout
 * 
 */
class AlertFrameFilter : public FrameFilter {                                           // <pyapi>

public:                                                                                 // <pyapi>
    AlertFrameFilter(const char *name, PyObject* pycallback, FrameFilter *next = NULL); // <pyapi>
    virtual ~AlertFrameFilter();                                                        // <pyapi>
protected:
    PyObject* pycallback;

protected:
    void run(Frame* frame);
    void go(Frame *frame);

};                                                                                      // <pyapi>


/** Dumps AVBitmapFrame(s) into a files
 * @ingroup filters_tag
 * 
 * You can convert the dumped yuv files into png images like this:
 * 
 * ffmpeg -pixel_format yuv420p -video_size 1920x1080 -i image_c10_s0_.yuv kokkelis.png
 * 
 */
class DumpAVBitmapFrameFilter : public FrameFilter { // <pyapi>

public:                                                          // <pyapi>
    DumpAVBitmapFrameFilter(const char *name, FrameFilter *next = NULL); // <pyapi>

protected:
    int count;

protected:
    void go(Frame *frame);
}; // <pyapi>



#endif
