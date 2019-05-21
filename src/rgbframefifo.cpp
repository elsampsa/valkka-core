/*
 * rgbframefifo.cpp :
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
 *  @file    rgbframefifo.cpp
 *  @author  Sampsa Riikonen
 *  @date    2019
 *  @version 0.11.0 
 *  
 *  @brief 
 */ 

#include "rgbframefifo.h"





RGBFrameFifo::RGBFrameFifo(RGBFrameFifoContext ctx) : FrameFifo("rgb_fifo", FrameFifoContext(0, 0, 0, 0, 20, 20, ctx.flush_when_full)) {
    int i;
    for(i=0; i < ctx.n; i++) {
        rgb_reservoir.push_back(new RGBFrame(ctx.max_width, ctx.max_height));
        rgb_stack.push_back(rgb_reservoir.back());
    }
}



RGBFrameFifo::~RGBFrameFifo() {
    for (auto it = rgb_reservoir.begin(); it != rgb_reservoir.end(); ++it) {
        delete *it;
    }
}


bool RGBFrameFifo::writeCopy(Frame* f, bool wait) {
    
    std::cout << "RGBFrameFifo : writeCopy : " << *f << std::endl;
    
    if (f->getFrameClass() != FrameClass::avrgb) {
        return FrameFifo::writeCopy(f, wait); // call motherclass "standard" writeCopy
    }
    
    {std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
    
        AVRGBFrame *avrgbframe = static_cast<AVRGBFrame*>(f);
        
        while (rgb_stack.empty()) { // deal with spurious wake-ups
            if (wait) {
                fifologger.log(LogLevel::normal) << "FrameFifo: "<<name<<" writeCopy: waiting for stack frames.  Frame="<<(*f)<<std::endl;
                this->ready_condition.wait(lk);
            }
            else {
                fifologger.log(LogLevel::fatal) << "FrameFifo: "<<name<<" writeCopy: OVERFLOW! No more frames in stack.  Frame="<<(*f)<<std::endl;
                if (ctx.flush_when_full) {
                    recycleAll_();
                }
                return false;
            }
        }
        
        RGBFrame *tmpframe = rgb_stack.front();  // .. the Frame* pointer to the Frame object is in reservoirs[FrameClass].  Frame*'s in stacks[FrameClass] are same Frame*'s as in reservoirs[FrameClass]
        rgb_stack.pop_front();                   // .. remove that pointer from the stack
    
        tmpframe->fromAVRGBFrame(avrgbframe);
        fifo.push_front(tmpframe);  // push_front takes a copy of the pointer // fifo: push: front, read: back
    
#ifdef FIFO_VERBOSE
        if (fifo.size()>1) {std::cout << "FrameFifo: "<<name<<" writeCopy: count=" << fifo.size() << std::endl;}
#endif
    
#ifdef TIMING_VERBOSE
        long int dt=(getCurrentMsTimestamp()-tmpframe->mstimestamp);
        if (dt>100) {
            std::cout << "FrameFifo: "<<name<<" writeCopy : timing : inserting frame " << dt << " ms late" << std::endl;
        }
#endif
    
        this->condition.notify_one(); // after receiving 
        return true;
    }
}


void RGBFrameFifo::recycle_(Frame* f) {// Return Frame f back into the stack.
    if (f->getFrameClass() != FrameClass::rgb) {
        FrameFifo::recycle_(f); // call motherclass "standard" recycle
        return;
    }
        
    RGBFrame *rgbframe = static_cast<RGBFrame*>(f);
  
    rgb_stack.push_back(rgbframe); // take: from the front.  recycle: to the back
}

/*
TODO:
- Test the fifo
- yolo module:
    - input frame filter : rgb filter => framefifo filter => rgbframefifo 
    - loop for reading rgbframefifo  .. just print the received frames & recycle them





*/
