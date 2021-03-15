/*
 * alsathread.cpp :
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
 *  @file    alsathread.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
 *  
 *  @brief 
 */ 

#include "alsathread.h"


ALSAThread(const char* name, FrameFifoContext ALSAThread::fifo_ctx=FrameFifoContext()) {
}

ALSAThread::~ALSAThread() {
    ALSASink *alsasink;
    
    alsalogger.log(LogLevel::crazy) << "ALSAThread: destructor: " << std::endl;
    
    stopCall(); // stop if not stopped ..
    for (std::vector<ALSASink*>::iterator it = slots_.begin(); it != slots_.end(); ++it) {
        alsasink=*it;
        if (!alsasink) {
        }
        else {
            delete alsasink;
        }
    }
}

void ALSAThread::run() {// Main execution loop
    long int mstime;
    long int old_mstime;
    
    Frame* f;
    long unsigned timeout;
    
    mstime=getCurrentMsTimestamp();
    old_mstime=mstime;
    loop=true;
    
    timeout=Timeout::alsathread;
    while(loop) {
        #ifdef PRESENT_VERBOSE
        std::cout << "ALSAThread "<< this->name <<" : run : timeout = " << timeout << std::endl;
        #endif
        f=infifo->read(timeout);
        if (!f) { // TIMEOUT : either one seconds has passed, or it's about time to present the next frame..
            if (debug) {
            }
            else {
                timeout=std::min(handleFifo(),Timeout::alsathread); // present/discard frames and return timeout to the last frame.  Recycles frames.  Returns the timeout
                // std::cout << "ALSAThread run : no frame : timeout : " << timeout << std::endl;
            }
        } // TIMEOUT
        else { // GOT FRAME // remember to apply infifo->recycle
            if (debug) {
                alsalogger.log(LogLevel::normal) << "ALSAThread "<< this->name <<" : run : DEBUG MODE! recycling received frame "<< *f << std::endl;
                infifo->recycle(f);
            }
            else {
                timeout=std::min(insertFifo(f),Timeout::alsathread); 
                // insert frame into the presentation fifo, get timeout to the last frame in the presentation fifo
                // ..frame is now in the presentation fifo.. remember to recycle on handleFifo
                while (timeout==0) {
                    timeout=std::min(handleFifo(),Timeout::alsathread);
                }
                #if defined(PRESENT_VERBOSE) || defined(TIMING_VERBOSE)
                // dumpFifo();
                std::cout << "ALSAThread " << this->name <<" : run : got frame : timeout : " <<timeout<<std::endl<<std::endl;
                #endif
            }
        } // GOT FRAME
        
        // time(&timer);
        mstime=getCurrentMsTimestamp();
        
        if ( (mstime-old_mstime)>=Timeout::alsathread ) {
            #ifdef PRESENT_VERBOSE
            std::cout << "ALSAThread " << this->name <<" : run : interrupt " << std::endl;
            #endif
            handleSignals();
            // checkSlots(mstime);
            old_mstime=mstime;
            #ifdef FIFO_DIAGNOSIS
            // diagnosis();
            #endif
        }
    }
}


long unsigned ALSAThread::insertFifo(Frame* f) {// sorted insert
    // handle special (signal) frames here
    if (f->getFrameClass() == FrameClass::signal) {
        SignalFrame *signalframe = static_cast<SignalFrame*>(f);
        handleSignal(signalframe->opengl_signal_ctx); // TODO: remove
        /*
        if (signalframe->signaltype==SignalType::alsathread) { // TODO
            handleSignal(signalframe->opengl_signal_ctx);
        }
        */
        // recycle(f); // alias
        infifo->recycle(f);
        return 0;
    }
    else if (f->getFrameClass() == FrameClass::setup) {
        alsalogger.log(LogLevel::debug) 
            << "ALSAThread insertFifo: SetupFrame: " << *f << std::endl;
        
        SetupFrame *setupframe = static_cast<SetupFrame*>(f);
        infifo->recycle(f);
        return 0;
    }
    
    /*
     *  timestamps in the presentation queue/fifo:
     *  
     *  <young                         old>
     *  90  80  70  60  50  40  30  20  10
     *  
     *  see also the comments in ALSAThread:handleFifo()
     *  
     *  Let's take a closer look at this .. "_" designates another camera, so here
     *  we have frames from two different cameras:
     *  
     *  90_ 85 80_ 71 70_  60_ 60 51 50_ 40 40_ 31 30_ 22_ 20  15_ 10
     *  
     *  incoming 39                            |
     */
    
    bool inserted=false;
    long int rel_mstimestamp;
    
    auto it=presfifo.begin();
    while(it!=presfifo.end()) {
        #ifdef PRESENT_VERBOSE // the presentation fifo will be extremely busy.. avoid unnecessary logging
        // std::cout << "ALSAThread insertFifo: iter: " << **it <<std::endl;
        #endif
        if (f->mstimestamp >= (*it)->mstimestamp ) {//insert before this element
            #ifdef PRESENT_VERBOSE
            //std::cout << "ALSAThread insertFifo: inserting "<< *f <<std::endl; // <<" before "<< **it <<std::endl;
            #endif
            presfifo.insert(it,f);
            inserted=true;
            break;
        }
        ++it;
    }
    if (!inserted) {
        #ifdef PRESENT_VERBOSE
        // std::cout << "ALSAThread insertFifo: inserting "<< *f <<" at the end"<<std::endl;
        #endif
        presfifo.push_back(f);
    }
    
    rel_mstimestamp=( presfifo.back()->mstimestamp-(getCurrentMsTimestamp()-msbuftime) );
    rel_mstimestamp=std::max((long int)0,rel_mstimestamp);
    
    if (rel_mstimestamp>future_ms_tolerance) { 
        // fifo might get filled up with frames too much in the future (typically wrong timestamps..) process them immediately
        #ifdef PRESENT_VERBOSE
        std::cout << "ALSAThread insertFifo: frame in distant future: "<< rel_mstimestamp <<std::endl;
        #endif
        rel_mstimestamp=0;
    }
    
    #ifdef PRESENT_VERBOSE
    std::cout << "ALSAThread insertFifo: returning timeout "<< rel_mstimestamp <<std::endl;
    #endif
    return (long unsigned)rel_mstimestamp; //return timeout to next frame
}


long unsigned ALSAThread::handleFifo() {// handles the presentation fifo
    // Check out the docs for the timestamp naming conventions, etc. in \ref timing
    long unsigned mstime_delta;         // == delta
    long int      rel_mstimestamp;      // == trel = t_ - (t-tb) = t_ - delta
    Frame*        f;                    // f->mstimestamp == t_
    bool          present_frame; 
    long int      mstime;  
    
    // mstime_delta=getCurrentMsTimestamp()-msbuftime; // delta = (t-tb)
    // mstime       =getCurrentMsTimestamp();
    // mstime_delta =mstime-msbuftime;
    
    #ifdef TIMING_VERBOSE
    resetCallTime();
    #endif
    
    auto it=presfifo.rbegin(); // reverse iterator
    
    while(it!=presfifo.rend()) {// while
        // mstime_delta=getCurrentMsTimestamp()-msbuftime; // delta = (t-tb)
        mstime       =getCurrentMsTimestamp();
        mstime_delta =mstime-msbuftime;
        
        f=*it; // f==pointer to frame
        rel_mstimestamp=(f->mstimestamp-mstime_delta); // == trel = t_ - delta
        if (rel_mstimestamp>0 and rel_mstimestamp<=future_ms_tolerance) {// frames from [inf,0) are left in the fifo
            // ++it;
            break; // in fact, just break the while loop (frames are in time order)
        }
        else {// remove the frame *f from the fifo.  Either scrap or present it
            // 40 20 -20 => found -20
            ++it; // go one backwards => 20 
            it= std::list<Frame*>::reverse_iterator(presfifo.erase(it.base())); // eh.. it.base() gives the next iterator (in forward sense?).. we'll remove that .. create a new iterator on the modded
            // it.base : takes -20
            // erase   : removes -20 .. returns 20
            // .. create a new reverse iterator from 20
            present_frame=false;
            
            #ifdef NO_LATE_DROP_DEBUG // present also the late frames
            // std::cout << "ALSAThread rel_mstimestamp, future_ms_tolerance : " << rel_mstimestamp << " " << future_ms_tolerance << std::endl;
            if (rel_mstimestamp>future_ms_tolerance) { // fifo might get filled up with future frames .. if they're too much in the future, scrap them
                alsalogger.log(LogLevel::normal) << "ALSAThread handleFifo: DISCARDING a frame too far in the future " << rel_mstimestamp << " " << *f << std::endl;
            } 
            else { // .. in all other cases, just present the frame
                present_frame=true;
            }
            
            #else
            if (rel_mstimestamp<=-10) {// scrap frames from [-10,-inf)
                // alsalogger.log(LogLevel::normal) << "ALSAThread handleFifo: DISCARDING late frame " << " " << rel_mstimestamp << " " << *f << std::endl;
                alsalogger.log(LogLevel::normal) << "ALSAThread handleFifo: DISCARDING late frame " << " " << rel_mstimestamp << " <" << f->mstimestamp <<"> " << std::endl;
            }
            else if (rel_mstimestamp>future_ms_tolerance) { // fifo might get filled up with future frames .. if they're too much in the future, scrap them
                alsalogger.log(LogLevel::normal) << "ALSAThread handleFifo: DISCARDING a frame too far in the future " << rel_mstimestamp << " " << *f << std::endl;
            }
            else if (rel_mstimestamp<=0) {// present frames from [0,-10)
                present_frame=true;
            }
            #endif
            
            if (present_frame) { // present_frame
                if (!slotOk(f->n_slot)) {//slot overflow, do nothing
                }
                else if (f->getFrameClass()==FrameClass::yuv) {// YUV FRAME
                    YUVFrame *yuvframe = static_cast<YUVFrame*>(f);
                    #if defined(PRESENT_VERBOSE) || defined(TIMING_VERBOSE)
                    std::cout<<"ALSAThread handleFifo: PRESENTING " << rel_mstimestamp << " <"<< yuvframe->mstimestamp <<"> " << std::endl;
                    if (it!=presfifo.rend()) {
                        std::cout<<"ALSAThread handleFifo: NEXT       " << (*it)->mstimestamp-mstime_delta << " <"<< (*it)->mstimestamp <<"> " << std::endl;
                    }
                    #endif
                    // if next frame was give too fast, scrap it
                    if (manageSlotTimer(yuvframe->n_slot,mstime)) {
                        activateSlotIf(yuvframe->n_slot, yuvframe); // activate if not already active
                        #ifdef TIMING_VERBOSE
                        reportCallTime(0);
                        #endif
                        // yuv_frame's pbos have already been uploaded to GPU.  Now they're loaded to the texture
                        // loadTEX uses slots_[], where each vector element is a SlotContext (=set of textures and a shader program)
                        loadYUVFrame(yuvframe->n_slot, yuvframe);
                        #ifdef TIMING_VERBOSE
                        reportCallTime(1);
                        #endif
                        render(yuvframe->n_slot); // renders all render groups that depend on this slot.  A slot => RenderGroups (x window) => list of RenderContext => SlotContext (textures)
                        #ifdef TIMING_VERBOSE
                        reportCallTime(2);
                        #endif
                    }
                    else {
                        // alsalogger.log(LogLevel::normal) << "ALSAThread handleFifo: feeding frames too fast! dropping.." << std::endl; // printed by manageSlotTimer => manageTimer
                    }
                } // YUV FRAME
            }// present frame
            infifo->recycle(f); // codec found or not, frame always recycled
        }// present or scrap
    }// while
    
    if (presfifo.empty()) {
        #ifdef PRESENT_VERBOSE
        std::cout<<"ALSAThread handleFifo: empty! returning default " << Timeout::alsathread << " ms timeout " << std::endl;
        #endif
        return Timeout::alsathread;
    }
    else {
        f=presfifo.back();
        mstime_delta=getCurrentMsTimestamp()-msbuftime; // delta = (t-tb)
        rel_mstimestamp=f->mstimestamp-mstime_delta; // == trel = t_ - delta
        rel_mstimestamp=std::max((long int)0,rel_mstimestamp);
        #ifdef PRESENT_VERBOSE
        std::cout<<"ALSAThread handleFifo: next frame: " << *f <<std::endl;
        std::cout<<"ALSAThread handleFifo: timeout   : " << rel_mstimestamp <<std::endl;
        #endif
        return (long unsigned)rel_mstimestamp; // time delay until the next presentation event..
    }
}


void ALSAThread::preRun() {
    
}

void ALSAThread::postRun() {
    
}

void ALSAThread::sendSignal(ALSASignalContext signal_ctx) {
    put_signal_context(&f, signal_ctx, SignalType::alsathread);
    infilter.run(&f);
}

void ALSAThread::requestStopCall() {
    alsalogger.log(LogLevel::crazy) << "ALSAThread: requestStopCall: "
        << this->name <<std::endl;
    if (!this->has_thread) { return; } // thread never started
    if (stop_requested) { return; } // can be requested only once
    stop_requested = true;
    
    ALSASignalContext signal_ctx;
    signal_ctx.signal=ALSASignal::exit;
    
    alsalogger.log(LogLevel::crazy) << "ALSAThread: sending exit signal "
        << this->name <<std::endl;
    this->sendSignal(signal_ctx);
}

void ALSAThread::handleSignal(ALSASignalContext &signal_ctx) {
    const ALSASignalPars &pars = signal_ctx.pars;

    switch (signal_ctx.signal) {
        case ALSASignal::exit:
            loop = false;
            break;

        case ALSASignal::new_playback:
            newPlaybackSink(pars.ALSAPlaybackContext);
            break;

        case ALSASignal::del_playback:
            delPlaybackSink(pars.ALSAPlaybackContext);
            break;
        }
}


int ALSAThread::safeGetSlot(SlotNumber slot, ALSASink*& alsasink_) {
    ALSASink* alsasink;
    alsalogger.log(LogLevel::crazy) 
        << "ALSAThread: safeGetSlot" << std::endl;
    
    if (slot>I_MAX_SLOTS) {
        alsalogger.log(LogLevel::fatal) 
            << "ALSAThread: safeGetSlot: WARNING! Slot number overfow : increase I_MAX_SLOTS in sizes.h" 
            << std::endl;
        return -1;
    }
    
    try {
        alsasink=this->slots_[slot];
    }
    catch (std::out_of_range) {
        alsalogger.log(LogLevel::debug) << "ALSAThread: safeGetSlot : slot " 
            << slot << " is out of range! " << std::endl;
        alsasink_=NULL;
        return -1;
    }
    if (!alsasink) {
        alsalogger.log(LogLevel::crazy) << "ALSAThread: safeGetSlot : nothing at slot " 
            << slot << std::endl;
        alsasink_=NULL;
        return 0;
    }
    else {
        alsalogger.log(LogLevel::debug) << "ALSAThread: safeGetSlot : returning " 
            << slot << std::endl;
        alsasink_=alsasink;
        return 1;
    }
}


void ALSAThread::newPlaybackSink(ALSAPlaybackContext& ctx) {

    switch (safeGetSlot(ctx.slot)) {
        case -1: // out of range
            break;
        case 0: // free
            ALSASinkContext sink_ctx 
                = ALSASinkContext(ctx.name, ctx.cardindex)
            this->slots_[ctx.slot] = 
                new ALSASink(sink_ctx);
            break;
        case 1: // reserved
            break;
    }
}

void ALSAThread::delPlaybackSink(ALSAPlaybackContext& ctx) {
    switch (safeGetSlot(ctx.slot)) {
        case -1: // out of range
            break;
        case 0: // free
            break;
        case 1: // reserved
            ALSASink *sink = this->slots_[ctx.slot];
            delete sink;
            this->slots_[ctx.slot] = NULL;
    }
}

/*
void ALSAThread::newRecordingContext(ALSARecordingContext& ctx) {
    switch (safeGetSlot(ctx.slot)) {
        case -1: // out of range
            break;
        case 0: // free
            ALSASourceContext source_ctx 
                = ALSASourceContext(ctx.name, ctx.cardindex)
            this->slots_source[ctx.slot] = 
                new ALSASource(source_ctx);
            break;
        case 1: // reserved
            break;
    }
}

void ALSAThread::delRecordingContext(ALSARecordingContext& ctx) {
    switch (safeGetSourceSlot(ctx.slot)) {
        case -1: // out of range
            break;
        case 0: // free
            break;
        case 1: // reserved
            ALSASource *source = this->slots_source[ctx.slot];
            delete sink;
            this->slots_source[ctx.slot] = NULL;
    }
}
*/

FifoFrameFilter ALSAThread::&getFrameFilter() {
    return &infifo;
}

void ALSAThread::newPlaybackSinkCall(ALSAPlaybackContext& ctx) {
    SignalFrame f = SignalFrame();
    ALSASignalContext signal_ctx = ALSASignalContext();

    // encapsulate parameters
    ALSASignalPars pars;
    ctx.id = std::rand();
    pars.playback_ctx = ctx;

    // encapsulate signal and parameters into signal context of the frame
    signal_ctx.signal = ALSASignal::new_playback;
    signal_ctx.pars   = pars;
    
    // f.custom_signal_ctx = (void*)(signal_ctx);
    put_signal_context(&f, signal_ctx, SignalType::alsathread);
    infilter.run(&f);
}

ALSAThread::delPlaybackSinkCall(ALSAPlaybackContext& ctx) {
    SignalFrame f = SignalFrame();
    ALSASignalContext signal_ctx = ALSASignalContext();

    // encapsulate parameters
    ALSASignalPars pars;
    pars.playback_ctx = ctx;

    // encapsulate signal and parameters into signal context of the frame
    signal_ctx.signal = ALSASignal::del_playback;
    signal_ctx.pars   = pars;
    
    // f.custom_signal_ctx = (void*)(signal_ctx);
    put_signal_context(&f, signal_ctx, SignalType::alsathread);
    infilter.run(&f);
}

/*
ALSAThread::newRecordingContextCall(ALSARecordingContext& ctx) {
    SignalFrame f = SignalFrame();
    ALSASignalContext signal_ctx = ALSASignalContext();

    // encapsulate parameters
    ALSASignalPars pars;
    pars.recording_ctx = ctx;
    ctx.id = std::rand();

    // encapsulate signal and parameters into signal context of the frame
    signal_ctx.signal = ALSASignal::new_recording;
    signal_ctx.pars   = pars;
    
    // f.custom_signal_ctx = (void*)(signal_ctx);
    put_signal_context(&f, signal_ctx, SignalType::alsathread);
    infilter.run(&f);    
    
}

ALSAThread::delRecordingContextCall(ALSARecordingContext& ctx) {
    SignalFrame f = SignalFrame();
    ALSASignalContext signal_ctx = ALSASignalContext();

    // encapsulate parameters
    ALSASignalPars pars;
    pars.recording_ctx = ctx;

    //TODO: set id

    // encapsulate signal and parameters into signal context of the frame
    signal_ctx.signal = ALSASignal::del_recording;
    signal_ctx.pars   = pars;
    
    // f.custom_signal_ctx = (void*)(signal_ctx);
    put_signal_context(&f, signal_ctx, SignalType::alsathread);
    infilter.run(&f);
}
*/

