/*
 * fdwritethread.cpp :
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
 *  @file    fdwritethread.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.0.3 
 *  
 *  @brief 
 */ 

#include "fdwritethread.h"

#define logger valkkafslogger


FDWrite::FDWrite(FrameFifo& fifo, const FDWriteContext& ctx) : fifo(fifo), ctx(ctx) {
}


FDWrite::~FDWrite() {
    for(auto it=internal_fifo.begin(); it!=internal_fifo.end(); ++it) {
        fifo.recycle(*it);
    }
}



FDWriteThread::FDWriteThread(const char* name, FrameFifoContext fifo_ctx) : Thread(name), infifo(name,fifo_ctx), infilter(name, &infifo), infilter_block(name, &infifo), write_fds(), read_fds(), timeout(msToTimeval(Timeout::fdwritethread)) {
    this->slots_.resize(I_MAX_SLOTS+1,NULL);
}
    
    
FDWriteThread::~FDWriteThread() {
    FDWrite *fd_write;    
    for (auto it = slots_.begin(); it != slots_.end(); ++it) {
        fd_write = *it;
        if (!fd_write) {
        }
        else {
            delete fd_write;
        }
    }
}
 
 
/*
FDWriterThread::run:
    
    map a slot to a file descriptor
    
                             +--[    ] fd = 5
                     fd=10   |
    FrameFilter --> [   ]----+--[||  ] fd = 6
                             |
                             +--[|   ] fd = 7
    
    
    thread loop:
    
        select([10], [6,7])
        => if (10) => update all queues => if others, use write => inspect queues, update lists
        commands come through fd 10
        
        
    read file-descriptor triggered
    => put frame to right queue = n .. add fd that corresponds to n into write-list
    
    write file-descriptor triggered
    => write stuff, inspect n .. if there's frames, add n's fd into write-list
        
    std::list<FDWrite*> writer_by_fd
    
    writer_by_fd[i]
    
*/



/*
void updateListsByWrite(int fd) {
    // manipulates select write list
    FDWrite *w = writer_by_fd[fd];
    // better to do a single iteration over writer_by_fd, instead of several calls to []
    
    if (w->internal_fifo.empty()) {
        writer_by_fd.pop(w);
    }
    else {
        // writer_by_fd[w] ==> into select write list
    }    
}


void updateListsByRead(int fd) {
    // manipulates select write list
    FDWrite *w = writer_by_fd[fd];
    
    if (w->internal_fifo.empty()) {
        writer_by_fd.pop(w);
    }
    else {
        // writer_by_fd[w] ==> into select write list
    }    
}
*/

void FDWriteThread::run() {
    loop = True;
    Frame *f;
    Frame *f2;
    FDWrite *fd_write_;
    int si;
    int read_fd = infifo.getFD();
    
    FD_ZERO(&read_fds);
    FD_ZERO(&write_fds);
    FD_SET(read_fd, &read_fds);
    
    while(loop) { // MAIN LOOP
        timeout = msToTimeval(Timeout::fdwritethread);
        
        si = select(nfds, &read_fds, &write_fds, NULL, &timeout);
        
        // (1) loop all writable fds .. write into them from frame queues of each FDWrite entry
        for(auto it = fd_writes.begin(); it != fd_writes.end(); ++it) { // WRITE FD LOOP
            if (FD_ISSET((*it)->ctx.fd, &write_fds)) { // CAN WRITE
                if ((*it)->internal_fifo.empty()) { 
                    // nothing to write ..
                }
                else {
                    // for valkka internal streaming, use BasicFrame::dump and raw_writer
                    // for muxed streams, use MuxedFrame::dump and file descriptor
                    // .. better to subclass.  Let's start with MuxFrame
                    f2 = (*it)->internal_fifo.back();
                    (*it)->internal_fifo.pop_back();
                    // TODO: this->sendFrame(f2); .. f2.dump((*it)->ctx.fd);
                    infifo.recycle(f2);
                }
            } // CAN WRITE
        } // WRITE FD LOOP
            
        
        // (2) read the incoming queue .. execute commands or push frame to a queue
        if (si == -1) {
            perror("FDWriterThread : run : select()");
        }
        else if (si) { // SELECT OK
        
            if (FD_ISSET(read_fd, &read_fds)) { // GOT FRAME // this must ALWAYS BE ACCOMPANIED WITH A RECYCLE CALL
                logger.log(LogLevel::crazy) << "FDWriteThread: got frame " << *f << std::endl;
                if (f->getFrameClass()==FrameClass::signal) { // SIGNALFRAME
                    SignalFrame *signalframe = static_cast<SignalFrame*>(f);
                    if (signalframe->signaltype==SignalType::fdwritethread) {
                        FDWriteSignalContext signal_ctx = FDWriteSignalContext();
                        get_signal_context(signalframe, signal_ctx);
                        handleSignal(signal_ctx);
                    }
                } // SIGNALFRAME
                else { // PAYLOAD FRAME
                    // BasicFrame or MuxFrame .. depends on the subclass
                    // TODO: this->queueFrame(f):
                    if (safeGetSlot(f->n_slot, fd_write_) > 0) {
                        fd_write_->internal_fifo.push_front(f); // TODO check allowed fifo size ==> FDWrite.pushIf()
                    }
                    
                } // PAYLOAD FRAME
                infifo.recycle(f);
                
            } // GOT FRAME
                
        } // SELECT OK
        else { // TIMEOUT
            logger.log(LogLevel::crazy) << "FDWriteThread: "<< this->name <<" timeout expired!" << std::endl;
        } // TIMEOUT
            
        // reconstruct read list
        nfds = read_fd;
        FD_SET(read_fd, &read_fds);
        
        // reconstruct the lists
        FD_ZERO(&read_fds);
        FD_ZERO(&write_fds);
        
        // reconstruct write list
        for(auto it = fd_writes.begin(); it != fd_writes.end(); ++it) { // WRITE FD LOOP
            if ((*it)->internal_fifo.empty()) { // write ready but nothing to write: remove this write desciptor
                // in the next round there will be nothing to write ..
            }
            else {
                nfds = std::max(nfds, (*it)->ctx.fd); // recalculate max file descriptor
                FD_SET((*it)->ctx.fd, &write_fds);
            }
        } // WRITE FD LOOP
        
    } // MAIN LOOP
}
    

void FDWriteThread::preRun() {
}

void FDWriteThread::postRun() {
}

void FDWriteThread::preJoin() {
}

void FDWriteThread::postJoin() {
}


void FDWriteThread::handleSignal(const FDWriteSignalContext &signal_ctx) {
    const FDWriteContext &ctx = signal_ctx.pars.fd_write_ctx;

    switch (signal_ctx.signal) {
        
        case FDWriteSignal::exit:
            loop = false;
            break;
        
        case FDWriteSignal::register_stream:
            registerStream(ctx);
            break;
        
        case FDWriteSignal::deregister_stream:
            deregisterStream(ctx);
            break;
        
    };
}


int FDWriteThread::safeGetSlot(const SlotNumber slot, FDWrite*& fd_write) { // -1 = out of range, 0 = free, 1 = reserved // &* = modify pointer in-place
    FDWrite* fd_write_;
    
    if (slot>I_MAX_SLOTS) {
        logger.log(LogLevel::fatal) << "FDWrite: safeGetSlot: WARNING! Slot number overfow : increase I_MAX_SLOTS in sizes.h" << std::endl;
        return -1;
    }
    
    try {
        fd_write_ = this->slots_[slot];
    }
    catch (std::out_of_range) {
        logger.log(LogLevel::debug) << "FDWrite: safeGetSlot : slot " << slot << " is out of range! " << std::endl;
        fd_write_ = NULL;
        return -1;
    }
    if (!fd_write_) {
        logger.log(LogLevel::crazy) << "FDWrite: safeGetSlot : nothing at slot " << slot << std::endl;
        fd_write = NULL;
        return 0;
    }
    else {
        logger.log(LogLevel::debug) << "FDWrite: safeGetSlot : returning " << slot << std::endl;
        fd_write = fd_write_;
        return 1;
    }
    
}


void FDWriteThread::setMaxFD() {
    nfds = infifo.getFD();
    for (auto it = fd_writes.begin(); it != fd_writes.end(); ++it) { // TODO: use fd_writes instead
        nfds = std::max(nfds, (*it)->ctx.fd);
    }
}


void FDWriteThread::registerStream(const FDWriteContext &ctx) {
    FDWrite* fd_write;
    
    logger.log(LogLevel::crazy) << "FDWriteThread: registerStream" << std::endl;
    switch (safeGetSlot(ctx.slot, fd_write)) {
        case -1: // out of range
            break;
            
        case 0: // slot is free
            this->slots_[ctx.slot] = new FDWrite(infifo, ctx); 
            FD_SET(ctx.fd, &write_fds);
            setMaxFD();
            logger.log(LogLevel::debug) << "FDWriteThread: registerStream : stream registered at slot " << ctx.slot << " with ptr " << this->slots_[ctx.slot] << std::endl;
            break;
            
        case 1: // slot is reserved
            logger.log(LogLevel::normal) << "FDWriteThread: registerStream : slot " << ctx.slot << " is reserved! " << std::endl;
            break;
    } // switch
}


void FDWriteThread::deregisterStream(const FDWriteContext &ctx) {
    FDWrite* fd_write;
    
    logger.log(LogLevel::crazy) << "FDWriteThread: deregisterStream" << std::endl;
    switch (safeGetSlot(ctx.slot, fd_write)) {
        case -1: // out of range
            break;
        case 0: // slot is free
            logger.log(LogLevel::crazy) << "FDWriteThread: deregisterStream : nothing at slot " << ctx.slot << std::endl;
            break;
        case 1: // slot is reserved
            logger.log(LogLevel::debug) << "FDWriteThread: deregisterStream : de-registering " << ctx.slot << std::endl;
            delete this->slots_[ctx.slot];
            this->slots_[ctx.slot] = NULL;
            FD_CLR(ctx.fd, &write_fds);
            setMaxFD();
            break;
    } // switch
}
    

void FDWriteThread::registerStreamCall(const FDWriteContext &ctx) {
    // context for sending the signal
    FDWriteSignalContext signal_ctx = FDWriteSignalContext();
    FDWriteSignalPars    pars;
    
    // signal parameters
    pars.fd_write_ctx = ctx;
    
    signal_ctx.signal = FDWriteSignal::register_stream;
    signal_ctx.pars   = pars;
    
    // prepare a signal frame
    SignalFrame f = SignalFrame();
    put_signal_context(&f, signal_ctx);
    
    // .. and send it to the queue
    infilter.run(&f);
}
    
    
void FDWriteThread::deregisterStreamCall(const FDWriteContext &ctx) {
    // context for sending the signal
    FDWriteSignalContext signal_ctx = FDWriteSignalContext();
    FDWriteSignalPars    pars;
    
    // signal parameters
    pars.fd_write_ctx = ctx;
    
    signal_ctx.signal = FDWriteSignal::deregister_stream;
    signal_ctx.pars   = pars;
    
    // prepare a signal frame
    SignalFrame f = SignalFrame();
    put_signal_context(&f, signal_ctx);
    
    // .. and send it to the queue
    infilter.run(&f);
}


void FDWriteThread::requestStopCall() {
    if (!this->has_thread) { return; } // thread never started
    if (stop_requested) { return; }    // can be requested only once
    stop_requested = true;

    FDWriteSignalContext signal_ctx = FDWriteSignalContext();
    FDWriteSignalPars    pars;
    
    // context for sending the signal
    signal_ctx.signal = FDWriteSignal::exit;
    signal_ctx.pars   = pars;
    
    // prepare a signal frame
    SignalFrame f = SignalFrame();
    put_signal_context(&f, signal_ctx);

    // .. and send it to the queue
    infilter.run(&f);
}


FifoFrameFilter& FDWriteThread::getFrameFilter() {
    return infilter;
}



