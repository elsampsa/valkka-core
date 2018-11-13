/*
 * valkkafsreader.cpp :
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
 *  @file    valkkafsreader.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
 *  
 *  @brief 
 */ 

#include "valkkafsreader.h"


ValkkaFSReaderThread::ValkkaFSReaderThread(const char *name, ValkkaFS &valkkafs, FrameFilter &outfilter, FrameFifoContext fifo_ctx) : Thread(name), valkkafs(valkkafs), outfilter(outfilter), infifo(name, fifo_ctx), infilter(name, &infifo), filestream(valkkafs.getDevice(), std::fstream::binary | std::fstream::in) {
}
    
    
ValkkaFSReaderThread::~ValkkaFSReaderThread() {
}
    
    
void ValkkaFSReaderThread::run() {
    Frame* f;
    time_t timer;
    time_t oldtimer;
    
    time(&timer);
    oldtimer=timer;
    loop=true;
    
    while(loop) {
        f=infifo.read(Timeout::valkkafsreaderthread);
        if (!f) { // TIMEOUT
            std::cout << "ValkkaFSReaderThread: "<< this->name <<" timeout expired!" << std::endl;
        }
        else { // GOT FRAME // this must ALWAYS BE ACCOMPANIED WITH A RECYCLE CALL
            // Handle signal frames
            if (f->getFrameClass()==FrameClass::signal) { // SIGNALFRAME
                SignalFrame *signalframe = static_cast<SignalFrame*>(f);
                handleSignal(signalframe->valkkafsreader_signal_ctx);
            } // SIGNALFRAME
            else {
                std::cout << "ValkkaFSWriterThread : " << this->name <<" accepts only SignalFrame " << std::endl;
            }
            infifo.recycle(f); // always recycle
        } // GOT FRAME
        
        time(&timer);
        
        // old-style ("interrupt") signal handling
        if ( (1000*difftime(timer,oldtimer)) >= Timeout::valkkafsreaderthread ) { // time to check the signals..
            handleSignals();
            oldtimer=timer;
        }
    }
}
    

void ValkkaFSReaderThread::preRun() {
}
    
void ValkkaFSReaderThread::postRun() {
}

void ValkkaFSReaderThread::handleSignal(ValkkaFSReaderSignalContext &signal_ctx) {
    switch (signal_ctx.signal) {
        
        case ValkkaFSReaderSignal::exit:
            loop=false;
            break;
            
        case ValkkaFSReaderSignal::set_slot_id:
            setSlotId(signal_ctx.pars.n_slot, signal_ctx.pars.id);
            break;
            
        case ValkkaFSReaderSignal::unset_slot_id:
            unSetSlotId(signal_ctx.pars.id);
            break;
            
        case ValkkaFSReaderSignal::clear_slot_id:
            clearSlotId();
            break;
            
        case ValkkaFSReaderSignal::report_slot_id:
            reportSlotId();
            break;
            
        case ValkkaFSReaderSignal::pull_blocks:
            pullBlocks(signal_ctx.pars.block_list);
            break;
            
        
    }
}

void ValkkaFSReaderThread::sendSignal(ValkkaFSReaderSignalContext signal_ctx) {
    std::unique_lock<std::mutex> lk(this->mutex);
    this->signal_fifo.push_back(signal_ctx);
}

void ValkkaFSReaderThread::handleSignals() {
    std::unique_lock<std::mutex> lk(this->mutex);
    // handle pending signals from the signals fifo
    for (auto it = signal_fifo.begin(); it != signal_fifo.end(); ++it) { // it == pointer to the actual object (struct SignalContext)
        handleSignal(*it);
    }
    signal_fifo.clear();
}

FifoFrameFilter &ValkkaFSReaderThread::getFrameFilter() {
    return infilter;
}


void ValkkaFSReaderThread::setSlotId(SlotNumber slot, IdNumber id) {
    std::cout << "ValkkaFSReaderThread: setSlotId: " << slot << " " << id << std::endl;
    auto it=id_to_slot.find(id);
    if (it==id_to_slot.end()) { // this slot does not exist
        id_to_slot.insert(std::make_pair(id, slot));
    }
    else {
        std::cout << "ValkkaFSReaderThread: setSlotId: id " << id << " reserved" << std::endl;
    }
}
    
void ValkkaFSReaderThread::unSetSlotId(IdNumber id) {
    std::cout << "ValkkaFSReaderThread: unSetSlotId: " << id << std::endl;
    auto it=id_to_slot.find(id);
    if (it==id_to_slot.end()) { // this slot does not exist
        std::cout << "ValkkaFSReaderThread: unSetSlotId: no such id " << id << std::endl;
    }
    else {
        id_to_slot.erase(it);
    }
}
    
void ValkkaFSReaderThread::clearSlotId() {
    std::cout << "ValkkaFSReaderThread: clearSlotId: " << std::endl;
    id_to_slot.clear();
}

void ValkkaFSReaderThread::reportSlotId() {
    std::cout << "ValkkaFSReaderThread: reportSlotId: " << std::endl;
    for(auto it=id_to_slot.begin(); it!=id_to_slot.end(); ++it) {
        std::cout << "ValkkaFSReaderThread: reportSlotId: " << it->first << " --> " << it->second << std::endl;
    }
}

void ValkkaFSReaderThread::pullBlocks(std::list<std::size_t> block_list) {
    IdNumber id;
    BasicFrame f = BasicFrame();
    
    for(auto it=block_list.begin(); it!=block_list.end(); it++) { // BLOCK LOOP
        std::cout << "ValkkaFSReaderThread : pullBlocks : " << *it << std::endl;
        if (*it < valkkafs.get_n_blocks()) { // BLOCK OK
            std::cout << "ValkkaFSReaderThread : pullBlocks : block seek " << valkkafs.getBlockSeek(*it) << std::endl;
            filestream.seekp(std::streampos(valkkafs.getBlockSeek(*it))); // TODO
            while(true) { // FRAME LOOP
                id = f.read(filestream);
                std::cout << "ValkkaFSReaderThread : pullBlocks : id " << id << std::endl;
                if (id==0) { // no more frames in this block
                    break;
                }
                else { // HAS FRAME
                    auto it2=id_to_slot.find(id);
                    if (it2==id_to_slot.end()) {
                        std::cout << "ValkkaFSReader: no slot for id " << id << std::endl;
                    }
                    else { // HAS SLOT
                        f.n_slot=it2->second;
                        outfilter.run(&f);
                        /*
                        bool seek = f.isSeekable();
                        std::cout << "[" << id << "] " << f;
                        if (seek) {
                            std::cout << " * ";
                        }
                        std::cout << std::endl;
                        */
                    } // HAS SLOT
                } // HAS FRAME
            } // FRAME LOOP
        } // BLOCK OK
    } // BLOCK LOOP
}

void ValkkaFSReaderThread::setSlotIdCall(SlotNumber slot, IdNumber id) {
    ValkkaFSReaderSignalContext signal_ctx;
    ValkkaFSReaderSignalPars    pars;
    
    pars.n_slot = slot;
    pars.id     = id;

    signal_ctx.signal = ValkkaFSReaderSignal::set_slot_id;
    signal_ctx.pars   = pars;
    
    // prepare a signal frame
    SignalFrame f = SignalFrame();
    f.valkkafsreader_signal_ctx = signal_ctx;
    // .. and send it to the queue
    infilter.run(&f);
}


void ValkkaFSReaderThread::unSetSlotIdCall(IdNumber id) {
    ValkkaFSReaderSignalContext signal_ctx;
    ValkkaFSReaderSignalPars    pars;
    
    pars.id = id;

    signal_ctx.signal = ValkkaFSReaderSignal::unset_slot_id;
    signal_ctx.pars   = pars;
    
    // prepare a signal frame
    SignalFrame f = SignalFrame();
    f.valkkafsreader_signal_ctx = signal_ctx;
    // .. and send it to the queue
    infilter.run(&f);
}

void ValkkaFSReaderThread::clearSlotIdCall() {
    ValkkaFSReaderSignalContext signal_ctx;
    ValkkaFSReaderSignalPars    pars;
    
    signal_ctx.signal = ValkkaFSReaderSignal::clear_slot_id;
    signal_ctx.pars   = pars;
    
    // prepare a signal frame
    SignalFrame f = SignalFrame();
    f.valkkafsreader_signal_ctx = signal_ctx;
    // .. and send it to the queue
    infilter.run(&f);
}

void ValkkaFSReaderThread::reportSlotIdCall() {
    ValkkaFSReaderSignalContext signal_ctx;
    signal_ctx.signal = ValkkaFSReaderSignal::report_slot_id;
    
    // prepare a signal frame
    SignalFrame f = SignalFrame();
    f.valkkafsreader_signal_ctx = signal_ctx;
    // .. and send it to the queue
    infilter.run(&f);
}


void ValkkaFSReaderThread::pullBlocksCall(std::list<std::size_t> block_list) {
    ValkkaFSReaderSignalContext signal_ctx;
    ValkkaFSReaderSignalPars    pars;
    
    pars.block_list = block_list;

    signal_ctx.signal = ValkkaFSReaderSignal::pull_blocks;
    signal_ctx.pars   = pars;
    
    // prepare a signal frame
    SignalFrame f = SignalFrame();
    f.valkkafsreader_signal_ctx = signal_ctx;
    // .. and send it to the queue
    infilter.run(&f);
}


void ValkkaFSReaderThread::pullBlocksPyCall(PyObject *pylist) {
    // pullBlocksCall(); // TODO
}


void ValkkaFSReaderThread::requestStopCall() {
    if (!this->has_thread) { return; } // thread never started
    if (stop_requested) { return; }    // can be requested only once
    stop_requested = true;

    // use the old-style "interrupt" way of sending signals
    ValkkaFSReaderSignalContext signal_ctx;
    signal_ctx.signal = ValkkaFSReaderSignal::exit;
    
    this->sendSignal(signal_ctx);
}
