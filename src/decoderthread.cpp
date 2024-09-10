/*
 * decoderthread.cpp :
 * 
 * (c) Copyright 2017-2024 Sampsa Riikonen
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
 *  @file    decoderthread.cpp
 *  @author  Sampsa Riikonen
 *  @date    2021
 *  @version 1.6.1 
 *  
 *  @brief 
 */ 

#include "decoderthread.h"
#include "logging.h"

//#define AVTHREAD_VERBOSE 1

DecoderThread::DecoderThread(const char* name, FrameFilter& outfilter, FrameFifoContext fifo_ctx) 
    : Thread(name), outfilter(outfilter), infifo(name,fifo_ctx), infilter(name,&infifo), 
    infilter_block(name,&infifo), is_decoding(false), mstimetolerance(0), n_threads(1),
    state(AbstractFileState::none), timefilter("av_thread_timestamp", &outfilter), use_time_correction(true)
    {
        avthreadlogger.log(LogLevel::debug) << "DecoderThread : constructor : N_MAX_DECODERS ="<<int(N_MAX_DECODERS)<<std::endl;
        decoders.resize(int(N_MAX_DECODERS),NULL);
        setupframes.resize(int(N_MAX_DECODERS));
    }


DecoderThread::~DecoderThread() {
    threadlogger.log(LogLevel::crazy) << "DecoderThread: destructor: "<< this->name <<std::endl;
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


Decoder* DecoderThread::chooseAudioDecoder(AVCodecID codec_id) {
    switch (codec_id) { // switch: audio codecs
        case AV_CODEC_ID_PCM_MULAW:
            return new DummyDecoder();
            break;
        default:
            return NULL;
            break;
    } // switch: audio codecs
}

Decoder* DecoderThread::chooseVideoDecoder(AVCodecID codec_id) {
    switch (codec_id) { // switch: video codecs
        case AV_CODEC_ID_H264:
            return new VideoDecoder(
                AV_CODEC_ID_H264, n_threads = this->n_threads);
            break;
        default:
            return NULL;
            break;        
    } // switch: video codecs
}

Decoder* DecoderThread::fallbackAudioDecoder(AVCodecID codec_id) {
    threadlogger.log(LogLevel::fatal) << "DecoderThread: no fallbackAudioDecoder"
        << std::endl;
    return NULL;
}


Decoder* DecoderThread::fallbackVideoDecoder(AVCodecID codec_id) {
    threadlogger.log(LogLevel::fatal) << "DecoderThread: no fallbackVideoDecoder"
        << std::endl;
    return NULL;
}

void DecoderThread::run() {
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
    /*
    long int mstimestamp, mstimestamp_old, mstimestamp_frame, mstimestamp_frame_old;

    mstimestamp = 0;
    mstimestamp_old = 0;
    mstimestamp_frame = 0;
    mstimestamp_frame_old = 0;
    */
    while(loop) {
        f=infifo.read(Timeout::avthread);
        if (!f) { // TIMEOUT
            #ifdef AVTHREAD_VERBOSE
            std::cout << "DecoderThread: "<< this->name <<" timeout expired!" << std::endl;
            #endif
        }
        else { // GOT FRAME // this must always be accompanied with a recycle call
            #ifdef AVTHREAD_VERBOSE
            std::cout << "DecoderThread: "<< this->name <<" got frame "<<*f << std::endl;
            #endif
            subsession_index=f->subsession_index;
            // info frame    : init decoder
            // regular frame : make a copy
            
            if (subsession_index >= n_decoders) { // got frame: subsession_index too big
                avthreadlogger.log(LogLevel::fatal) << "DecoderThread: "<< this->name <<" : run : decoder slot overlow : "<<subsession_index<<"/"<<n_decoders<< std::endl; // we can only have that many decoder for one stream
                infifo.recycle(f); // return frame to the stack - never forget this!
            } // got frame: subsession_index too big
            
            else if (f->getFrameClass()==FrameClass::setup) { // got frame : SETUP
                SetupFrame *setupframe = static_cast<SetupFrame*>(f);
                
                // avthreadlogger.log(LogLevel::debug) << "DecoderThread: " << name << " got SetupFrame: " << *setupframe << std::endl;
                
                if (setupframe->sub_type == SetupFrameType::stream_init) { // SETUP: STREAM INIT
                    setupframes[subsession_index].copyFrom(setupframe); // cache the setupframe
                    if (decoders[subsession_index]!=NULL) { // slot is occupied
                        avthreadlogger.log(LogLevel::debug) << "DecoderThread: "<< this->name <<" : run : decoder reinit " << std::endl;
                        delete decoders[subsession_index];
                        decoders[subsession_index]=NULL;
                    }
                    
                    // register a new decoder
                    avthreadlogger.log(LogLevel::debug) << "DecoderThread: "<< this->name <<" : run : registering decoder for subsession " <<subsession_index<< std::endl;
                    
                    if (setupframe->media_type==AVMEDIA_TYPE_AUDIO) { // AUDIO
                        decoders[subsession_index]=this->chooseAudioDecoder(setupframe->codec_id);
                    } // AUDIO
                    else if (setupframe->media_type==AVMEDIA_TYPE_VIDEO) { // VIDEO
                        decoders[subsession_index]=this->chooseVideoDecoder(setupframe->codec_id);
                    } // VIDEO
                    else { // UNKNOW MEDIA TYPE
                        avthreadlogger.log(LogLevel::fatal) << "DecoderThread: "<< this->name <<" : run : unknown media type " << std::endl;
                        decoders[subsession_index]=new DummyDecoder();
                    }
                } // SETUP STREAM INIT
                
                else if (setupframe->sub_type == SetupFrameType::stream_state) { // SETUP STREAM STATE
                    
                    // std::cout << "DecoderThread: setupframe: stream_state: " << *f << std::endl;
                    
                    outfilter.run(f); // pass the setupframe downstream (they'll go all the way to OpenGLThread)
                    if (setupframe->stream_state == AbstractFileState::seek) {
                        // go into seek state .. don't forward frames
                        state = AbstractFileState::seek;
                        avthreadlogger.log(LogLevel::debug) << "DecoderThread: " << name << " setupframe: seek mode " << *f << std::endl;
                    }
                    else if (setupframe->stream_state == AbstractFileState::play) {
                        state = AbstractFileState::play;
                    }
                    else if (setupframe->stream_state == AbstractFileState::stop) { // typically, marks end of seek
                        avthreadlogger.log(LogLevel::debug) << "DecoderThread: " << name << " setupframe: stop mode " << *f << std::endl;
                        state = AbstractFileState::stop;
                        // during seek, frames, starting from the latest I-frame to the required frame, are decoded in a sprint
                        // because of this, the frames typically arrive late to the final destination (say, OpenGLThread)
                        // so we take the last decoded frame and send it again with a corrected timestamp (current time)
                        if (decoder and decoder->hasFrame()) {
                            Frame *tmpf = decoder->output();
                            tmpf->mstimestamp = getCurrentMsTimestamp();
                            avthreadlogger.log(LogLevel::debug) << "DecoderThread: " << name << " resend: " << *((AVMediaFrame*)tmpf) << std::endl;
                            outfilter.run(tmpf);
                            decoder->releaseOutput();
                        }
                    }
                    else {
                    }
                } // SETUP STREAM STATE
                infifo.recycle(f); // return frame to the stack - never forget this!
            } // got frame : SETUP
            
            else if (decoders[subsession_index]==NULL) { // woops, no decoder registered yet..
                avthreadlogger.log(LogLevel::debug) << "DecoderThread: "<< this->name <<" : run : no decoder registered for stream " << subsession_index << std::endl;
                infifo.recycle(f); // return frame to the stack - never forget this!
            }
            
            else if (f->getFrameClass()==FrameClass::none) { // void frame, do nothing // TODO: BasicFrames are decoded, but some other frame classes might be passed as-is through the decoder (just copy them and pass)
                infifo.recycle(f); // return frame to the stack - never forget this!
            }
            
            else if (f->getFrameClass()==FrameClass::basic) { // basic payload frame
                BasicFrame *basicframe = static_cast<BasicFrame*>(f);
                
                if (is_decoding) { // IS DECODING
                    decoder=decoders[subsession_index]; // alias
                    // Take a local copy of the frame, return the original to the stack, and then start (the time consuming) decoding
                    // decoder->in_frame.copyFrom(basicframe); // deep copy of the BasicFrame.  After performing the copy ..
                    decoder->input(basicframe); // decoder takes as an input a Frame and makes a copy of it.  Decoder must check that it is the correct type
                    infifo.recycle(f); // .. return frame to the stack
                    // infifo->dumpStack();
                    if (decoder->pull()) { // DECODER HAS STUFF
                        #ifdef AVTHREAD_VERBOSE
                        std::cout << "DecoderThread: "<< this->name <<" : run : decoder num " <<subsession_index<< " got frame " << std::endl;
                        #endif
                        
                        #ifdef PROFILE_TIMING
                        dt=(getCurrentMsTimestamp()-decoder->getMsTimestamp());
                        // std::cout << "[PROFILE_TIMING] DecoderThread: " << this->name <<" run: decoder sending frame at " << dt << " ms" << std::endl;
                        if (dt>=10) {
                            std::cout << "[PROFILE_TIMING] DecoderThread: " << this->name <<" run: decoder sending frame " << dt << " ms late" << std::endl;
                        }
                        #endif
                        if (mstimetolerance>0) { // late frames can be dropped here, before their insertion to OpenGLThreads fifo
                            if ((getCurrentMsTimestamp()-decoder->getMsTimestamp())<=mstimetolerance) {
                                // outfilter.run(&(decoder->out_frame));
                                outfilter.run(decoder->output()); // returns a reference to a decoded frame
                                //..the frame returned by decoder->output() runs through the whole filterchain until it has been copied in 
                                // the end of the filterchain (this typically ends up in another framefifo that creates a copy of the frame)
                                decoder->releaseOutput(); // in some cases - if the decoder does some internal buffering - we want to inform the decoder
                                // that the frame can now be overwritten etc.
                            }
                            else {
                                avthreadlogger.log(LogLevel::debug) << "DecoderThread: " << name << " : not sending late frame " << *(decoder->output()) << std::endl;
                                decoder->releaseOutput();
                            }
                        }
                        else { // no time tolerance defined
                            //outfilter.run(&(decoder->out_frame));
                            if (state != AbstractFileState::seek) { // frames during seek are discarded
                                // outfilter.run(decoder->output()); // returns a reference to a decoded frame
                                // std::cout << "decoder feeding" << std::endl;
                                if (use_time_correction) {
                                    #ifdef AVTHREAD_VERBOSE
                                    std::cout << "DecoderThread: "<< this->name <<" : run : sending time-corrected frame downstream" << std::endl;
                                    #endif
                                    timefilter.run(decoder->output());
                                    decoder->releaseOutput();
                                }
                                else {
                                    #ifdef AVTHREAD_VERBOSE
                                    std::cout << "DecoderThread: "<< this->name <<" : run : sending frame downstream" << std::endl;
                                    #endif
                                    outfilter.run(decoder->output());
                                    decoder->releaseOutput();
                                }
                            }
                            else {
                                avthreadlogger.log(LogLevel::debug) << "DecoderThread: seek: " << name << " scrapping frame: " << *(decoder->output()) << std::endl;
                                decoder->releaseOutput();
                            }
                            
                            #ifdef PROFILE_TIMING
                            dt=(getCurrentMsTimestamp()-decoder->getMsTimestamp());
                            // std::cout << "[PROFILE_TIMING] DecoderThread: " << this->name <<" run: decoder sending frame at " << dt << " ms" << std::endl;
                            if (dt>=10) {
                                std::cout << "[PROFILE_TIMING] DecoderThread: " << this->name <<" run: AFTER sending frame " << dt << " ms late" << std::endl;
                                // during the filterchain run, YUV frame is uploaded to GPU
                            }
                            #endif
                        }

                        /*
                        mstimestamp = getCurrentMsTimestamp();
                        std::cout << "Decoder: dt " << mstimestamp - mstimestamp_old << std::endl;
                        mstimestamp_old = mstimestamp;

                        mstimestamp_frame = decoder->output()->mstimestamp;
                        std::cout << "Decoder: frame dt " << mstimestamp_frame - mstimestamp_frame_old << std::endl;
                        mstimestamp_frame_old = mstimestamp_frame;
                        */


                    } // DECODER HAS STUFF
                    else {
                        #ifdef AVTHREAD_VERBOSE
                        std::cout << "DecoderThread: "<< this->name <<" : run : decoder num " <<subsession_index<< " no frame " << std::endl;
                        #endif
                    }

                    if (!decoder->isOk()) { // DECODER GONE SOUR
                        avthreadlogger.log(LogLevel::fatal) << "DecoderThread: a sour decoder.  Will try to get a fallback one" << std::endl;
                        delete decoders[subsession_index];
                        SetupFrame *setupframe = &setupframes[subsession_index];
                        if (setupframe->media_type==AVMEDIA_TYPE_AUDIO) { // AUDIO
                            decoders[subsession_index]=this->fallbackAudioDecoder(setupframe->codec_id);
                        } // AUDIO
                        else if (setupframe->media_type==AVMEDIA_TYPE_VIDEO) { // VIDEO
                            decoders[subsession_index]=this->fallbackVideoDecoder(setupframe->codec_id);
                        } // VIDEO
                        else { // UNKNOW MEDIA TYPE
                            decoders[subsession_index]=new DummyDecoder();
                        }
                        if (!decoders[subsession_index]) {
                            avthreadlogger.log(LogLevel::fatal) << "DecoderThread: sooo sour.  No fallback decoder either" << std::endl;
                            //hmm.. maybe should crash the whole program here..?
                        }
                    } // DECODER GONE SOUR
                    else {
                        // avthreadlogger.log(LogLevel::normal) << "DecoderThread: decoder still ok" << std::endl;
                    }

                } // IS DECODING
                else { // not decoding ..
                    infifo.recycle(f);      // .. return frame to the stack
                }
            } // basic payload frame
            
            else { // all other possible cases ..
                //std::cout << "DecoderThread: wtf?" << std::endl;
                infifo.recycle(f);
            }
            
        } // GOT FRAME
        
        mstime = getCurrentMsTimestamp();
        dt = mstime-oldmstime;
        // old-style ("interrupt") signal handling
        if (dt>=Timeout::avthread) { // time to check the signals..
            // avthreadlogger.log(LogLevel::verbose) << "DecoderThread: run: interrupt, dt= " << dt << std::endl;
            handleSignals();
            oldmstime=mstime;
            #ifdef FIFO_DIAGNOSIS
            infifo.diagnosis();
            #endif
        }
        
    }
}


void DecoderThread::preRun() {
    avthreadlogger.log(LogLevel::debug) << "DecoderThread: "<< name << " : preRun " << std::endl;
    ffmpeg_av_register_all(); // TODO: do this elsewhere!
}


void DecoderThread::postRun() {
    avthreadlogger.log(LogLevel::debug) << "DecoderThread: "<< name << " : postRun " << std::endl;
}


FifoFrameFilter &DecoderThread::getFrameFilter() {
    return infilter;
}


FifoFrameFilter &DecoderThread::getBlockingFrameFilter() {
    return (FifoFrameFilter&)infilter_block;
}


void DecoderThread::setTimeTolerance(long int mstol) {
    mstimetolerance=mstol;
}

void DecoderThread::setTimeCorrection(bool val) {
    use_time_correction = val;
}


FrameFifo &DecoderThread::getFifo() {
    return infifo;
}


void DecoderThread::sendSignal(AVSignalContext signal_ctx) {
    std::unique_lock<std::mutex> lk(this->mutex);
    this->signal_fifo.push_back(signal_ctx);
}


void DecoderThread::handleSignals() {
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

void DecoderThread::setNumberOfThreads(int n_threads) {
    this->n_threads = n_threads;
}

void DecoderThread::decodingOnCall() {
    AVSignalContext signal_ctx;
    signal_ctx.signal=AVSignal::on;
    sendSignal(signal_ctx);
}


void DecoderThread::decodingOffCall() {
    AVSignalContext signal_ctx;
    signal_ctx.signal=AVSignal::off;
    sendSignal(signal_ctx);
}


/*
 * void DecoderThread::stopCall() {
 *  threadlogger.log(LogLevel::debug) << "DecoderThread: stopCall: "<< this->name <<std::endl;
 *  if (!this->has_thread) {return;}
 *  AVSignalContext signal_ctx;
 *  signal_ctx.signal=AVSignal::exit;
 *  sendSignal(signal_ctx);
 *  this->closeThread();
 *  this->has_thread=false;
 * }
 */


void DecoderThread::requestStopCall() {
    threadlogger.log(LogLevel::crazy) << "DecoderThread: requestStopCall: "<< this->name <<std::endl;
    if (!this->has_thread) { return; } // thread never started
    if (stop_requested) { return; } // can be requested only once
    stop_requested = true;
    
    AVSignalContext signal_ctx;
    signal_ctx.signal = AVSignal::exit;
    
    threadlogger.log(LogLevel::crazy) << "DecoderThread: sending exit signal "<< this->name <<std::endl;
    this->sendSignal(signal_ctx);
}

