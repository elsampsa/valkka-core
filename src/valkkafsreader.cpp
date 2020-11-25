/*
 * valkkafsreader.cpp :
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
 *  @file    valkkafsreader.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.0.2 
 *  
 *  @brief 
 */ 

#include "valkkafsreader.h"


ValkkaFSReaderThread::ValkkaFSReaderThread(const char *name, ValkkaFS &valkkafs, FrameFilter &outfilter, FrameFifoContext fifo_ctx, bool o_direct) : Thread(name), valkkafs(valkkafs), outfilter(outfilter), infifo(name, fifo_ctx), infilter(name, &infifo), raw_reader(valkkafs.getDevice().c_str(), o_direct) {
}
    
    
ValkkaFSReaderThread::~ValkkaFSReaderThread() {
    raw_reader.close_();
}
    
    
void ValkkaFSReaderThread::run() {
    Frame* f;
    long int dt=0;
    long int mstime, oldmstime;

    mstime = getCurrentMsTimestamp();
    oldmstime = mstime;
    
    loop=true;
    
    while(loop) {
        f=infifo.read(Timeout::valkkafsreaderthread);
        if (!f) { // TIMEOUT
            valkkafslogger.log(LogLevel::crazy) <<"ValkkaFSReaderThread: "<< this->name <<" timeout expired!" << std::endl;
        }
        else { // GOT FRAME // this must ALWAYS BE ACCOMPANIED WITH A RECYCLE CALL
            // Handle signal frames
            if (f->getFrameClass()==FrameClass::signal) { // SIGNALFRAME
                SignalFrame *signalframe = static_cast<SignalFrame*>(f);
                handleSignal(signalframe->valkkafsreader_signal_ctx);
            } // SIGNALFRAME
            else {
                valkkafslogger.log(LogLevel::debug) <<"ValkkaFSWriterThread : " << this->name <<" accepts only SignalFrame " << std::endl;
            }
            infifo.recycle(f); // always recycle
        } // GOT FRAME
        
        mstime = getCurrentMsTimestamp();
        dt = mstime-oldmstime;
        // old-style ("interrupt") signal handling
        if (dt>=Timeout::valkkafsreaderthread) { // time to check the signals..
            // valkkafslogger.log(LogLevel::crazy) <<"ValkkaFSReaderThread: run: interrupt, dt= " << dt << std::endl;
            handleSignals();
            oldmstime=mstime;
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
    valkkafslogger.log(LogLevel::debug) <<"ValkkaFSReaderThread: setSlotId: " << slot << " " << id << std::endl;
    auto it=id_to_slot.find(id);
    if (it==id_to_slot.end()) { // this slot does not exist
        id_to_slot.insert(std::make_pair(id, slot));
    }
    else {
        valkkafslogger.log(LogLevel::debug) <<"ValkkaFSReaderThread: setSlotId: id " << id << " reserved" << std::endl;
    }
}
    
void ValkkaFSReaderThread::unSetSlotId(IdNumber id) {
    valkkafslogger.log(LogLevel::debug) <<"ValkkaFSReaderThread: unSetSlotId: " << id << std::endl;
    auto it=id_to_slot.find(id);
    if (it==id_to_slot.end()) { // this slot does not exist
        valkkafslogger.log(LogLevel::debug) <<"ValkkaFSReaderThread: unSetSlotId: no such id " << id << std::endl;
    }
    else {
        id_to_slot.erase(it);
    }
}
    
void ValkkaFSReaderThread::clearSlotId() {
    valkkafslogger.log(LogLevel::debug) <<"ValkkaFSReaderThread: clearSlotId: " << std::endl;
    id_to_slot.clear();
}

void ValkkaFSReaderThread::reportSlotId() {
    std::cout <<"ValkkaFSReaderThread: reportSlotId: " << std::endl;
    for(auto it=id_to_slot.begin(); it!=id_to_slot.end(); ++it) {
        std::cout <<"ValkkaFSReaderThread: reportSlotId: " << it->first << " --> " << it->second << std::endl;
    }
}

void ValkkaFSReaderThread::pullBlocks(std::list<std::size_t> block_list) {
    IdNumber id;
    BasicFrame f = BasicFrame();
    MarkerFrame start_marker = MarkerFrame();
    MarkerFrame end_marker = MarkerFrame();
    
    start_marker.tm_start=true;
    end_marker.tm_end=true;
    
    for(auto it=block_list.begin(); it!=block_list.end(); it++) { // BLOCK LOOP
        valkkafslogger.log(LogLevel::debug) <<"ValkkaFSReaderThread : pullBlocks : " << *it << std::endl;
        if (*it < valkkafs.get_n_blocks()) { // BLOCK OK
            valkkafslogger.log(LogLevel::debug) <<"ValkkaFSReaderThread : pullBlocks : block seek " << valkkafs.getBlockSeek(*it) << std::endl;
            // filestream.seekp(std::streampos(valkkafs.getBlockSeek(*it))); // TODO
            raw_reader.seek(valkkafs.getBlockSeek(*it));
            
            while(true) { // FRAME LOOP
                // id = f.read(filestream);
                id = f.read(raw_reader);
                
                // valkkafslogger.log(LogLevel::debug) <<"ValkkaFSReaderThread : pullBlocks : id " << id << std::endl;
                if (id==0) { // no more frames in this block
                    break;
                }
                else { // HAS FRAME
                    if (start_marker.mstimestamp==0) { // mark transmission start
                        start_marker.mstimestamp=f.mstimestamp;
                        outfilter.run(&start_marker);
                    }
                    auto it2=id_to_slot.find(id);
                    if (it2==id_to_slot.end()) {
                        valkkafslogger.log(LogLevel::debug) <<"ValkkaFSReader: no slot for id " << id << std::endl;
                    }
                    else { // HAS SLOT
                        f.n_slot=it2->second;
                        f.fillPars();
                        // std::cout << "ValkkaFSReader: transmitting " << f << std::endl;
                        outfilter.run(&f);
                        /*
                        bool seek = f.isSeekable();
                        valkkafslogger.log(LogLevel::normal) <<"[" << id << "] " << f;
                        if (seek) {
                            valkkafslogger.log(LogLevel::normal) <<" * ";
                        }
                        valkkafslogger.log(LogLevel::normal) <<std::endl;
                        */
                    } // HAS SLOT
                } // HAS FRAME
            } // FRAME LOOP
        } // BLOCK OK
    } // BLOCK LOOP
    if (start_marker.mstimestamp==0) { // mark transmission start (there were no frames)
        start_marker.mstimestamp=f.mstimestamp;
        outfilter.run(&start_marker);
    }
    end_marker.mstimestamp=f.mstimestamp; // can be zero as well
    outfilter.run(&end_marker);
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
    Py_INCREF(pylist);
    
    valkkafslogger.log(LogLevel::debug) << "ValkkaFSReaderThread: pullBlocksPyCall" << std::endl;
    if (!PyList_Check(pylist)) {
        valkkafslogger.log(LogLevel::fatal) << "ValkkaFSReaderThread: pullBlocksPyCall: not a python list" << std::endl;
        return;
    }
    std::list<std::size_t> block_list;
    PyObject *element;
    // Py_ssize_t PyList_Size(PyObject *list)
    Py_ssize_t i;
    for(i=0; i<PyList_Size(pylist); i++) {
        element=PyList_GetItem(pylist,i);
        if (PyLong_Check(element)) {
            block_list.push_back(PyLong_AsSize_t(element));
        }
    }
    valkkafslogger.log(LogLevel::debug) << "ValkkaFSReaderThread: pullBlocksPyCall: pylist= ";
    for(auto it=block_list.begin(); it!=block_list.end(); it++) {
        valkkafslogger.log(LogLevel::debug) <<*it << " ";
    }
    valkkafslogger.log(LogLevel::debug) <<std::endl;
    
    pullBlocksCall(block_list);
    Py_DECREF(pylist);
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
