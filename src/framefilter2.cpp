/*
 * framefilter2.cpp :
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
 *  @file    framefilter2.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
 *  
 *  @brief 
 */ 

#include "framefilter2.h"


AlertFrameFilter::AlertFrameFilter(const char *name, 
    PyObject* pycallback, FrameFilter *next) : FrameFilter(name, next) {
    if (PyCallable_Check(pycallback) == 1) {
        this->pycallback = pycallback;
    }
    else {
        filterlogger.log(LogLevel::fatal) << "AlertFrameFilter: setCallback: could not set callback" << std::endl;
    }
}

void AlertFrameFilter::go(Frame* frame) {
}

void AlertFrameFilter::run(Frame* frame) {
    bool pass = true;

    if (frame->getFrameClass()==FrameClass::signal) {    
        SignalFrame *signalframe = static_cast<SignalFrame*>(frame);
        if (signalframe->signaltype==SignalType::offline) {
            pass = false;
            OfflineSignalContext ctx = OfflineSignalContext();
            get_signal_context(signalframe, ctx);

            filterlogger.log(LogLevel::debug) << "AlertFrameFilter: run: got alert" << std::endl;

            // delete signal_ctx; // custom signal context must be freed // NOPES! since it's not necessarily caugt donwstream
            if (pycallback != NULL) { // PYCALLBACK
                PyGILState_STATE gstate;
                gstate = PyGILState_Ensure();
                
                PyObject *res, *tup;

                tup = PyTuple_New(1);
                PyTuple_SET_ITEM(tup, 1, 
                    PyLong_FromUnsignedLong((unsigned long)(ctx.n_slot)));
                
                res = PyObject_CallFunctionObjArgs(pycallback, tup, NULL);
                // send a tuple with : (slot)
                if (!res) {
                    filterlogger.log(LogLevel::fatal) << "AlertFrameFilter: alert callback failed" << std::endl;
                }
                PyGILState_Release(gstate);
            } // PYCALLBACK 
        } // offline signal
    } // signal

    if (pass) {
        if (!this->next)
        {
            return;
        } // call next filter .. if there is any
        (this->next)->run(frame);
    }
}

