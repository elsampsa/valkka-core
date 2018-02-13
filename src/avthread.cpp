/*
 * avthread.cpp : FFmpeg decoding thread
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
 *  @file    avthread.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.0 
 *  @brief   FFmpeg decoding thread
 */ 

#include "avthread.h"
#include "logging.h"
#include "tools.h"

// AVThread::AVThread(const char* name, FrameFifo* infifo, FrameFilter& outfilter) : Thread(name), infifo(infifo), outfilter(outfilter), is_decoding(false) {
AVThread::AVThread(const char* name, FrameFifo& infifo, FrameFilter& outfilter, int core_id, long int mstimetolerance) : Thread(name, core_id), infifo(infifo), outfilter(outfilter), mstimetolerance(mstimetolerance), is_decoding(false) {
  avthreadlogger.log(LogLevel::debug) << "AVThread : constructor : N_MAX_DECODERS ="<<int(N_MAX_DECODERS)<<std::endl;
  decoders.resize(int(N_MAX_DECODERS),NULL);
}


AVThread::~AVThread() {
  DecoderBase* decoder;
  for (std::vector<DecoderBase*>::iterator it = decoders.begin(); it != decoders.end(); ++it) {
  decoder=*it;
  if (!decoder) {
    }
  else {
    delete decoder;
    }
 }
}



void AVThread::run() {
  bool ok;
  bool got_frame;
  unsigned short subsession_index;
  Frame* f;
  time_t timer;
  time_t oldtimer;
  DecoderBase* decoder; // alias
  long int dt;
  
  time(&timer);
  oldtimer=timer;
  loop=true;
  
  start_mutex.unlock();
  while(loop) {
    // f=infifo->read(1000);
    f=infifo.read(Timeouts::avthread);
    if (!f) { // TIMEOUT
#ifdef AVTHREAD_VERBOSE
      std::cout << "AVThread: "<< this->name <<" timeout expired!" << std::endl;
#endif
      }
    else { // GOT FRAME // this must always be accompanied with a recycle call
#ifdef AVTHREAD_VERBOSE
      std::cout << "AVThread: "<< this->name <<" got frame "<<*f << std::endl;
#endif
      subsession_index=f->subsession_index;
      // info frame    : init decoder
      // regular frame : make a copy
      if (subsession_index>=decoders.size()) { // got frame: subsession_index too big
        avthreadlogger.log(LogLevel::fatal) << "AVThread: "<< this->name <<" : run : decoder slot overlow : "<<subsession_index<<"/"<<decoders.size()<< std::endl; // we can only have that many decoder for one stream
        infifo.recycle(f); // return frame to the stack - never forget this!
      }
      else if (f->frametype==FrameType::setup) { // got frame: DECODER INIT
        if (decoders[subsession_index]!=NULL) { // slot is occupied
          avthreadlogger.log(LogLevel::debug) << "AVThread: "<< this->name <<" : run : decoder reinit " << std::endl;
          delete decoders[subsession_index];
          decoders[subsession_index]=NULL;
        }
        // register a new decoder
        avthreadlogger.log(LogLevel::debug) << "AVThread: "<< this->name <<" : run : registering decoder to slot " <<subsession_index<< std::endl;
        switch ( (f->setup_pars).frametype ) { // NEW_CODEC_DEV // when adding new codecs, make changes here: add relevant decoder per codec
          case FrameType::h264:
            decoders[subsession_index]=new VideoDecoder(AV_CODEC_ID_H264);
            break;
          case FrameType::pcmu:
            decoders[subsession_index]=new DummyDecoder();
            break;
          default:
            decoders[subsession_index]=new DummyDecoder();
            break;
        } // switch
        infifo.recycle(f); // return frame to the stack - never forget this!
      } // got frame: DECODER INIT
      else if (decoders[subsession_index]==NULL) { // woops, no decoder registered yet..
        avthreadlogger.log(LogLevel::normal) << "AVThread: "<< this->name <<" : run : no decoder registered for stream " << subsession_index << std::endl;
        infifo.recycle(f); // return frame to the stack - never forget this!
      }
      else if (f->frametype==FrameType::none) { // void frame, do nothing
        infifo.recycle(f); // return frame to the stack - never forget this!
      }
      else if (is_decoding) { // decode
        decoder=decoders[subsession_index]; // alias
        // Take a local copy of the frame, return the original to the stack, and then start (the time consuming) decoding
        decoder->in_frame = *f; // deep copy of the frame.  After performing the copy ..
        infifo.recycle(f);      // .. return frame to the stack
        // infifo->dumpStack();
        if (decoder->pull()) { // of course, frame must be returned to stack and stack must be released, before doing anything time consuming (like decoding..)
#ifdef AVTHREAD_VERBOSE
          std::cout << "AVThread: "<< this->name <<" : run : decoder num " <<subsession_index<< " got frame " << std::endl;
#endif
          
#ifdef OPENGL_TIMING
          dt=(getCurrentMsTimestamp()-decoder->out_frame.mstimestamp);
          if (dt>=500) {
            std::cout << "AVThread: " << this->name <<" run: timing : decoder sending frame " << dt << " ms late" << std::endl;
          }
#endif
          if (mstimetolerance>0) { // late frames can be dropped here, before their insertion to OpenGLThreads fifo
            if ((getCurrentMsTimestamp()-decoder->out_frame.mstimestamp)<=mstimetolerance) {
              outfilter.run(&(decoder->out_frame));
            }
            else {
              avthreadlogger.log(LogLevel::debug) << "AVThread: not sending late frame " << decoder->out_frame << std::endl;
            }
          }
          else { // no time tolerance defined
            outfilter.run(&(decoder->out_frame));
          }
        }
        else {
#ifdef AVTHREAD_VERBOSE
          std::cout << "AVThread: "<< this->name <<" : run : decoder num " <<subsession_index<< " no frame " << std::endl;
#endif
        }
      } // decode
      else { // some other case .. what that might be?
        infifo.recycle(f);      // .. return frame to the stack
      }
      
    } // GOT FRAME
    
    time(&timer);
    
    if (difftime(timer,oldtimer)>=1) { // time to check the signals..
      handleSignals();
      oldtimer=timer;
#ifdef FIFO_DIAGNOSIS
      infifo.diagnosis();
#endif
    }
    
  }
}


void AVThread::preRun() {
  avthreadlogger.log(LogLevel::debug) << "AVThread: "<< name << " : preRun " << std::endl;
  ffmpeg_av_register_all(); // TODO: do this elsewhere!
}


void AVThread::postRun() {
  avthreadlogger.log(LogLevel::debug) << "AVThread: "<< name << " : postRun " << std::endl;
}


void AVThread::sendSignal(SignalContext signal_ctx) {
  std::unique_lock<std::mutex> lk(this->mutex);
  this->signal_fifo.push_back(signal_ctx);
}


void AVThread::handleSignals() {
  std::unique_lock<std::mutex> lk(this->mutex);
  // AVConnectionContext connection_ctx;
  unsigned short int i;
  
  if (signal_fifo.empty()) {return;}
  
  // handle pending signals from the signals fifo
  for (std::deque<SignalContext>::iterator it = signal_fifo.begin(); it != signal_fifo.end(); ++it) { // it == pointer to the actual object (struct SignalContext)
    
    switch (it->signal) {
      case Signals::exit:
        loop=false;
        break;
      case Signals::on: // start decoding
        is_decoding=true;
        break;
      case Signals::off:  // stop decoding
        is_decoding=false;
        break;
      }
  }
    
  signal_fifo.clear();
}

// API

void AVThread::decodingOnCall() {
  SignalContext signal_ctx;
  signal_ctx.signal=Signals::on;
  sendSignal(signal_ctx);
}


void AVThread::decodingOffCall() {
  SignalContext signal_ctx;
  signal_ctx.signal=Signals::off;
  sendSignal(signal_ctx);
}
  

void AVThread::stopCall() {
  SignalContext signal_ctx;
  
  signal_ctx.signal=Signals::exit;
  sendSignal(signal_ctx);
  this->closeThread();
  this->has_thread=false;
}

