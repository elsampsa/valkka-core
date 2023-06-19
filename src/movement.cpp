/*
 * movement.cpp : Framefilter implementing a movement detector
 * 
 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
 * 
 * Authors: Petri Eranko <petri.eranko@dasys.fi>
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
 *  @file    movement.cpp
 *  @author  Petri Eranko
 *  @date    2019
 *  @version 1.3.6 
 *  
 *  @brief 
 */ 

#include "movement.h"

// #define MOVEMENTFILTER_DEBUG 1


MovementFrameFilter::MovementFrameFilter(const char* name, long int interval, float treshold, long int duration, FrameFilter* next) : FrameFilter(name, next), interval(interval), treshold(treshold), duration(duration), pycallback(NULL), movement_start(0), luma(NULL), mstimestamp(0), y_size(0) {
}

MovementFrameFilter::~MovementFrameFilter() {
    this->reset();
}

void MovementFrameFilter::setCallback(PyObject* pycallback) {
    if (PyCallable_Check(pycallback) == 1) {
        this->pycallback = pycallback;
    }
    else {
        std::cout << "MovementFrameFilter: setCallback: could not set callback" << std::endl;
    }
}


void MovementFrameFilter::reset() {
    mstimestamp = 0;
    if (luma) {
        delete[] luma;
    }
    luma = NULL;
    movement_start = 0;
    y_size = 0;
}


void MovementFrameFilter::go(Frame* frame) {
}
    
  
void MovementFrameFilter::run(Frame* frame) {
    if (frame->getFrameClass()!=FrameClass::avbitmap) {
        filterlogger.log(LogLevel::fatal) << "MovementFrameFilter: wrong kind of frame" << std::endl;
        return;
    }
    
    AVBitmapFrame *f = static_cast<AVBitmapFrame*>(frame);
    
    if (mstimestamp == 0) {
        // no saved previous timestamp yet ..
        mstimestamp = f->mstimestamp;
        return;
    }
    
    if ( (f->mstimestamp - mstimestamp) < interval) { // not enough time passed yet ..
        return;
    }
    
#ifdef MOVEMENTFILTER_DEBUG
    std::cout << "MovementFrameFilter: going forward: dt: " << (f->mstimestamp - mstimestamp) << std::endl;
#endif
    
    mstimestamp = f->mstimestamp; // time to update the timestamp ..

    if ( (luma == NULL) or (y_size != f->bmpars.y_size) ) { // time to re-allocate the luma plane
        if (luma) {
            delete[] luma;
        }
        y_size = f->bmpars.y_size;
        luma = new uint8_t[y_size];
        memcpy(luma, f->y_payload, y_size);
        return; // next time we'll have that cached frame ..
    }
    
    // compare new frame and cached frame
    int i;
    uint su;
    su = 0;
    for(i = 0; i < y_size; i++) {
        if (luma[i] >= f->y_payload[i]) {
            su += uint(luma[i] - f->y_payload[i]);
        }
        else {
            su += uint(f->y_payload[i] - luma[i]);
        }
    }
    
    memcpy(luma, f->y_payload, y_size); // cache the new frame
    float res = float(su) / float(y_size) / float(255); // normalize
    
#ifdef MOVEMENTFILTER_DEBUG
    std::cout << "MovementFrameFilter: result: " << res << std::endl;
#endif
    
    if (res >= treshold) {
        #ifdef MOVEMENTFILTER_DEBUG
        std::cout << "MovementFrameFilter: movement" << std::endl;
        #endif
        if (movement_start == 0) { // new movement event
            #ifdef MOVEMENTFILTER_DEBUG
            std::cout << "MovementFrameFilter: new movement event" << std::endl;
            #endif
            movement_start = mstimestamp;
            
            if (pycallback != NULL) { // PYCALLBACK
                PyGILState_STATE gstate;
                gstate = PyGILState_Ensure();
                
                PyObject *res, *tup;
            
                // res = Py_BuildValue("ll", (long int)(1), (long int)(2));
                // res = Py_BuildValue("(ii)", 123, 456); // debug : crassssh
                
                tup = PyTuple_New(3);
                PyTuple_SET_ITEM(tup, 0, Py_True);
                PyTuple_SET_ITEM(tup, 1, PyLong_FromUnsignedLong((unsigned long)(f->n_slot))); // unsigned short
                PyTuple_SET_ITEM(tup, 2, PyLong_FromLong(mstimestamp));
                
                res = PyObject_CallFunctionObjArgs(pycallback, tup, NULL);
                // send a tuple with : (True, slot, mstimestamp)
                if (!res) {
                    filterlogger.log(LogLevel::fatal) << "MovementFrameFilter: movement callback failed" << std::endl;
                }
                PyGILState_Release(gstate);
            } // PYCALLBACK
        }
        next->run(frame); // pass the frame
    }
    else if ( movement_start > 0 and ( (mstimestamp - movement_start) >= duration ) ) { // no movement detected, but old movement event was on and max event duration is due
        movement_start = 0;
        #ifdef MOVEMENTFILTER_DEBUG
        std::cout << "MovementFrameFilter: movement stopped" << std::endl;
        #endif

        if (pycallback != NULL) { // PYCALLBACK
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();
            
            PyObject *res, *tup;
        
            // res = Py_BuildValue("ll", (long int)(1), (long int)(2));
            // res = Py_BuildValue("(ii)", 123, 456); // debug : crassssh
            
            tup = PyTuple_New(3);
            PyTuple_SET_ITEM(tup, 0, Py_False);
            PyTuple_SET_ITEM(tup, 1, PyLong_FromUnsignedLong((unsigned long)(f->n_slot))); // unsigned short
            PyTuple_SET_ITEM(tup, 2, PyLong_FromLong(mstimestamp));
            
            res = PyObject_CallFunctionObjArgs(pycallback, tup, NULL);
            // send a tuple with : (True, slot, mstimestamp)
            if (!res) {
                filterlogger.log(LogLevel::fatal) << "MovementFrameFilter: movement callback failed" << std::endl;
            }
            PyGILState_Release(gstate);
        } // PYCALLBACK
    }
    else if ( movement_start > 0 ) { // movement event is on
        next->run(frame); // pass the frame
    }
    
}

