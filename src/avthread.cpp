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
 *  @version 0.14.0 
 *  @brief   FFmpeg decoding thread
 */ 

#include "avthread.h"
#include "logging.h"


AVThread::AVThread(const char* name, FrameFilter& outfilter, FrameFifoContext fifo_ctx) : Thread(name), outfilter(outfilter), infifo(name,fifo_ctx), infilter(name,&infifo), infilter_block(name,&infifo), is_decoding(false), mstimetolerance(0), state(AbstractFileState::none), n_threads(1) {
    avthreadlogger.log(LogLevel::debug) << "AVThread : constructor : N_MAX_DECODERS ="<<int(N_MAX_DECODERS)<<std::endl;
    decoders.resize(int(N_MAX_DECODERS),NULL);
}


AVThread::~AVThread() {
    threadlogger.log(LogLevel::crazy) << "AVThread: destructor: "<< this->name <<std::endl;
    stopCall();
    Decoder* decoder;
    for (std::vector<Decoder*>::iterator it = decoders.begin(); it != decoders.end(); ++it) {
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
    int subsession_index;
    Frame* f;
    Decoder* decoder = NULL; // alias

    long int dt=0;
    long int mstime, oldmstime;
    
    mstime = getCurrentMsTimestamp();
    oldmstime = mstime;
    loop=true;
    
    int n_decoders = int(decoders.size()); // std::size_t to int
    
    while(loop) {
        f=infifo.read(Timeout::avthread);
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
            
            if (subsession_index >= n_decoders) { // got frame: subsession_index too big
                avthreadlogger.log(LogLevel::fatal) << "AVThread: "<< this->name <<" : run : decoder slot overlow : "<<subsession_index<<"/"<<n_decoders<< std::endl; // we can only have that many decoder for one stream
                infifo.recycle(f); // return frame to the stack - never forget this!
            } // got frame: subsession_index too big
            
            else if (f->getFrameClass()==FrameClass::setup) { // got frame : SETUP
                SetupFrame *setupframe = static_cast<SetupFrame*>(f);
                
                // avthreadlogger.log(LogLevel::debug) << "AVThread: " << name << " got SetupFrame: " << *setupframe << std::endl;
                
                if (setupframe->sub_type == SetupFrameType::stream_init) { // SETUP: STREAM INIT
                
                    if (decoders[subsession_index]!=NULL) { // slot is occupied
                        avthreadlogger.log(LogLevel::debug) << "AVThread: "<< this->name <<" : run : decoder reinit " << std::endl;
                        delete decoders[subsession_index];
                        decoders[subsession_index]=NULL;
                    }
                    
                    // register a new decoder
                    avthreadlogger.log(LogLevel::debug) << "AVThread: "<< this->name <<" : run : registering decoder for subsession " <<subsession_index<< std::endl;
                    
                    if (setupframe->media_type==AVMEDIA_TYPE_AUDIO) { // AUDIO
                        
                        switch (setupframe->codec_id) { // switch: audio codecs
                            case AV_CODEC_ID_PCM_MULAW:
                                decoders[subsession_index]=new DummyDecoder();
                                break;
                            default:
                                break;
                                
                        } // switch: audio codecs
                    } // AUDIO
                    else if (setupframe->media_type==AVMEDIA_TYPE_VIDEO) { // VIDEO
                        
                        switch (setupframe->codec_id) { // switch: video codecs
                            case AV_CODEC_ID_H264:
                                decoders[subsession_index]=new VideoDecoder(AV_CODEC_ID_H264, n_threads = this->n_threads);
                                break;
                            default:
                                break;
                                
                        } // switch: video codecs
                        
                    } // VIDEO
                    else { // UNKNOW MEDIA TYPE
                        decoders[subsession_index]=new DummyDecoder();
                    }
                } // SETUP STREAM INIT
                
                else if (setupframe->sub_type == SetupFrameType::stream_state) { // SETUP STREAM STATE
                    
                    // std::cout << "AVThread: setupframe: stream_state: " << *f << std::endl;
                    
                    outfilter.run(f); // pass the setupframe downstream (they'll go all the way to OpenGLThread)
                    if (setupframe->stream_state == AbstractFileState::seek) {
                        // go into seek state .. don't forward frames
                        state = AbstractFileState::seek;
                        avthreadlogger.log(LogLevel::debug) << "AVThread: " << name << " setupframe: seek mode " << *f << std::endl;
                    
                    }
                    else if (setupframe->stream_state == AbstractFileState::play) {
                        state = AbstractFileState::play;
                    }
                    else if (setupframe->stream_state == AbstractFileState::stop) { // typically, marks end of seek
                        avthreadlogger.log(LogLevel::debug) << "AVThread: " << name << " setupframe: stop mode " << *f << std::endl;
                        state = AbstractFileState::stop;
                        // during seek, frames, starting from the latest I-frame to the required frame, are decoded in a sprint
                        // because of this, the frames typically arrive late to the final destination (say, OpenGLThread)
                        // so we take the last decoded frame and send it again with a corrected timestamp (current time)
                        if (decoder and decoder->hasFrame()) {
                            Frame *tmpf = decoder->output();
                            tmpf->mstimestamp = getCurrentMsTimestamp();
                            avthreadlogger.log(LogLevel::debug) << "AVThread: " << name << " resend: " << *((AVMediaFrame*)tmpf) << std::endl;
                            outfilter.run(tmpf);
                        }
                    }
                    else {
                    }
                } // SETUP STREAM STATE
                infifo.recycle(f); // return frame to the stack - never forget this!
            } // got frame : SETUP
            
            else if (decoders[subsession_index]==NULL) { // woops, no decoder registered yet..
                avthreadlogger.log(LogLevel::debug) << "AVThread: "<< this->name <<" : run : no decoder registered for stream " << subsession_index << std::endl;
                infifo.recycle(f); // return frame to the stack - never forget this!
            }
            
            else if (f->getFrameClass()==FrameClass::none) { // void frame, do nothing // TODO: BasicFrames are decoded, but some other frame classes might be passed as-is through the decoder (just copy them and pass)
                infifo.recycle(f); // return frame to the stack - never forget this!
            }
            
            else if (f->getFrameClass()==FrameClass::basic) { // basic payload frame
                BasicFrame *basicframe = static_cast<BasicFrame*>(f);
                
                if (is_decoding) { // is decoding
                    decoder=decoders[subsession_index]; // alias
                    // Take a local copy of the frame, return the original to the stack, and then start (the time consuming) decoding
                    // decoder->in_frame.copyFrom(basicframe); // deep copy of the BasicFrame.  After performing the copy ..
                    decoder->input(basicframe); // decoder takes as an input a Frame and makes a copy of it.  Decoder must check that it is the correct type
                    infifo.recycle(f); // .. return frame to the stack
                    // infifo->dumpStack();
                    if (decoder->pull()) { // decode
                        #ifdef AVTHREAD_VERBOSE
                        std::cout << "AVThread: "<< this->name <<" : run : decoder num " <<subsession_index<< " got frame " << std::endl;
                        #endif
                        
                        #ifdef PROFILE_TIMING
                        dt=(getCurrentMsTimestamp()-decoder->getMsTimestamp());
                        // std::cout << "[PROFILE_TIMING] AVThread: " << this->name <<" run: decoder sending frame at " << dt << " ms" << std::endl;
                        if (dt>=300) {
                            std::cout << "[PROFILE_TIMING] AVThread: " << this->name <<" run: decoder sending frame " << dt << " ms late" << std::endl;
                        }
                        #endif
                        if (mstimetolerance>0) { // late frames can be dropped here, before their insertion to OpenGLThreads fifo
                            if ((getCurrentMsTimestamp()-decoder->getMsTimestamp())<=mstimetolerance) {
                                // outfilter.run(&(decoder->out_frame));
                                outfilter.run(decoder->output()); // returns a reference to a decoded frame
                            }
                            else {
                                avthreadlogger.log(LogLevel::debug) << "AVThread: " << name << " : not sending late frame " << *(decoder->output()) << std::endl;
                            }
                        }
                        else { // no time tolerance defined
                            //outfilter.run(&(decoder->out_frame));
                            if (state != AbstractFileState::seek) { // frames during seek are discarded
                                outfilter.run(decoder->output()); // return a reference to a decoded frame
                            }
                            else {
                                avthreadlogger.log(LogLevel::debug) << "AVThread: seek: " << name << " scrapping frame: " << *(decoder->output()) << std::endl;
                            }
                            
                            #ifdef PROFILE_TIMING
                            dt=(getCurrentMsTimestamp()-decoder->getMsTimestamp());
                            // std::cout << "[PROFILE_TIMING] AVThread: " << this->name <<" run: decoder sending frame at " << dt << " ms" << std::endl;
                            if (dt>=300) {
                                std::cout << "[PROFILE_TIMING] AVThread: " << this->name <<" run: AFTER sending frame " << dt << " ms late" << std::endl;
                                // during the filterchain run, YUV frame is uploaded to GPU
                            }
                            #endif
                        }
                    } // decode
                    else {
                        #ifdef AVTHREAD_VERBOSE
                        std::cout << "AVThread: "<< this->name <<" : run : decoder num " <<subsession_index<< " no frame " << std::endl;
                        #endif
                    }
                } // is decoding
                else { // not decoding ..
                    infifo.recycle(f);      // .. return frame to the stack
                }
            } // basic payload frame
            
            else { // all other possible cases ..
                std::cout << "AVThread: wtf?" << std::endl;
                infifo.recycle(f);
            }
            
        } // GOT FRAME
        
        mstime = getCurrentMsTimestamp();
        dt = mstime-oldmstime;
        // old-style ("interrupt") signal handling
        if (dt>=Timeout::avthread) { // time to check the signals..
            // avthreadlogger.log(LogLevel::verbose) << "AVThread: run: interrupt, dt= " << dt << std::endl;
            handleSignals();
            oldmstime=mstime;
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


FifoFrameFilter &AVThread::getFrameFilter() {
    return infilter;
}


FifoFrameFilter &AVThread::getBlockingFrameFilter() {
    return (FifoFrameFilter&)infilter_block;
}


void AVThread::setTimeTolerance(long int mstol) {
    mstimetolerance=mstol;
}


FrameFifo &AVThread::getFifo() {
    return infifo;
}


void AVThread::sendSignal(AVSignalContext signal_ctx) {
    std::unique_lock<std::mutex> lk(this->mutex);
    this->signal_fifo.push_back(signal_ctx);
}


void AVThread::handleSignals() {
    std::unique_lock<std::mutex> lk(this->mutex);
    // AVConnectionContext connection_ctx;
    unsigned short int i;
    
    // if (signal_fifo.empty()) {return;}
    
    // handle pending signals from the signals fifo
    for (auto it = signal_fifo.begin(); it != signal_fifo.end(); ++it) { // it == pointer to the actual object (struct SignalContext)
        
        switch (it->signal) {
            case AVSignal::exit:
                loop=false;
                break;
            case AVSignal::on: // start decoding
                is_decoding=true;
                break;
            case AVSignal::off:  // stop decoding
                is_decoding=false;
                break;
        }
    }
    
    signal_fifo.clear();
}

// API


void AVThread::setNumberOfThreads(int n_threads) {
    this->n_threads = n_threads;
}

void AVThread::decodingOnCall() {
    AVSignalContext signal_ctx;
    signal_ctx.signal=AVSignal::on;
    sendSignal(signal_ctx);
}


void AVThread::decodingOffCall() {
    AVSignalContext signal_ctx;
    signal_ctx.signal=AVSignal::off;
    sendSignal(signal_ctx);
}


/*
 * void AVThread::stopCall() {
 *  threadlogger.log(LogLevel::debug) << "AVThread: stopCall: "<< this->name <<std::endl;
 *  if (!this->has_thread) {return;}
 *  AVSignalContext signal_ctx;
 *  signal_ctx.signal=AVSignal::exit;
 *  sendSignal(signal_ctx);
 *  this->closeThread();
 *  this->has_thread=false;
 * }
 */


void AVThread::requestStopCall() {
    threadlogger.log(LogLevel::crazy) << "AVThread: requestStopCall: "<< this->name <<std::endl;
    if (!this->has_thread) { return; } // thread never started
    if (stop_requested) { return; } // can be requested only once
    stop_requested = true;
    
    AVSignalContext signal_ctx;
    signal_ctx.signal = AVSignal::exit;
    
    threadlogger.log(LogLevel::crazy) << "AVThread: sending exit signal "<< this->name <<std::endl;
    this->sendSignal(signal_ctx);
}
