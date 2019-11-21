/*
 * muxer.cpp : FFmpeg muxers, implemented as Valkka framefilters
 * 
 * Copyright 2017, 2018, 2019 Valkka Security Ltd. and Sampsa Riikonen.
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
 *  @file    muxer.cpp
 *  @author  Sampsa Riikonen
 *  @date    2019
 *  @version 0.14.0 
 *  
 *  @brief 
 */ 

#include "muxer.h"

#define logger filterlogger //TODO: create a new logger for muxers

// #define MUXPARSE  //enable if you need to see what the byte parser is doing


MuxFrameFilter::MuxFrameFilter(const char* name, FrameFilter *next) : FrameFilter(name, next), active(false), initialized(false), mstimestamp0(0), zerotimeset(false), ready(false), av_format_ctx(NULL), avio_ctx(NULL), avio_ctx_buffer(NULL), missing(0), ccf(0), av_dict(NULL), format_name("matroska") {
    // two substreams per stream
    this->codec_contexes.resize(2,NULL);
    this->streams.resize(2, NULL);

    this->setupframes.resize(2);
    this->timebase = av_make_q(1,1000); // we're using milliseconds
    this->avpkt = new AVPacket();
    av_init_packet(this->avpkt); 
    
    this->avio_ctx_buffer = (uint8_t*)av_malloc(this->avio_ctx_buffer_size);
    this->avio_ctx = avio_alloc_context(this->avio_ctx_buffer, this->avio_ctx_buffer_size, 1, this, this->read_packet, this->write_packet, this->seek); // no read, nor seek
        
    /*
    // subclasses define format & muxing parameters
    // For example, fragmented MP4:
    
    format_name = std::string("mp4")

    // -movflags empty_moov+omit_tfhd_offset+frag_keyframe+separate_moof -frag_size
    av_dict_set(&av_dict, "movflags", "empty_moov+omit_tfhd_offset+frag_keyframe+separate_moof", 0);

    // av_dict_set(&av_dict, "frag_size", "500", 500); // nopes
    av_dict_set(&av_dict, "frag_size", "512", 0);
    */
}


MuxFrameFilter::~MuxFrameFilter() {
    deActivate();
    av_free(avio_ctx_buffer);
    av_free_packet(avpkt);
    delete avpkt;
    av_free(avio_ctx);
    av_dict_free(&av_dict);
}


void MuxFrameFilter::initMux() {
    int i;
    AVCodecID codec_id;
    initialized = false;
    missing = 0;

    this->defineMux(); // fills av_dict & format_name
    
    // create output context, open files
    i = avformat_alloc_output_context2(&av_format_context, NULL, format_name.c_str(), NULL);
    
    if (!av_format_context) {
        logger.log(LogLevel::fatal) << "MuxFrameFilter : initMux : FATAL : could not create output context!  Have you registered codecs and muxers? " << std::endl;
        avformat_free_context(av_format_context);
        exit(2);
    }

    // *** custom IO *** // https://www.ffmpeg.org/doxygen/3.2/avformat_8h_source.html
    av_format_context->pb = avio_ctx;
    av_format_context->flags |= AVFMT_FLAG_CUSTOM_IO;
    av_format_context->flags |= AVFMT_FLAG_FLUSH_PACKETS;
    av_format_context->flags |= AVFMT_FLAG_NOBUFFER;
    av_format_context->flags |= AVFMT_FLAG_NONBLOCK;
    av_format_context->flags |= AVFMT_FLAG_NOFILLIN;
    av_format_context->flags |= AVFMT_FLAG_NOPARSE;
    
    // *** normal file IO *** (for comparison / debugging)
    // i = avio_open(&av_format_context->pb, "paska", AVIO_FLAG_WRITE);
    avformat_init_output(av_format_context, &av_dict);
    
    // use the saved setup frames (if any) to set up the streams
    // Frame *frame; // alias
    for (auto it = setupframes.begin(); it != setupframes.end(); it++) {
        SetupFrame &setupframe = *it;
        if (setupframe.subsession_index < 0) { // not been initialized
        }
        else { // got setupframe
            AVCodecID codec_id = setupframe.codec_id;
                
            if (codec_id != AV_CODEC_ID_NONE) {
                AVCodecContext *av_codec_context;
                AVStream       *av_stream;
                
                av_codec_context = avcodec_alloc_context3(avcodec_find_decoder(codec_id));
                av_codec_context->width = N720.width; // dummy values .. otherwise mkv muxer refuses to co-operate
                av_codec_context->height = N720.height;
                av_codec_context->bit_rate = 1024*1024;
                
                av_codec_context->time_base.num = 1;
                av_codec_context->time_base.den = 20; // fps
                
                av_stream = avformat_new_stream(av_format_context, av_codec_context->codec); // av_codec_context->codec == AVCodec (i.e. we create a stream having a certain codec)
                
                av_stream->time_base = av_codec_context->time_base;
                // av_stream->codec->codec_tag = 0;

                av_stream->id = setupframe.subsession_index;
                
                /*
                // write some reasonable values here.  I'm unable to re-write this .. should be put into av_codec_context ?
                AVRational fps = AVRational();
                fps.num = 20;
                fps.den = 1;
                av_stream->avg_frame_rate = fps;
                */
                i = avcodec_parameters_from_context(av_stream->codecpar, av_codec_context);
                
                // std::cout << "MuxFrameFilter : initMux : context and stream " << std::endl;
                codec_contexes[setupframe.subsession_index] = av_codec_context;
                streams[setupframe.subsession_index] = av_stream;
                initialized = true; // so, at least one substream init'd
            }
        } // got setupframe
    }

    if (!initialized) {
        return;
    }

    ///*
    // codec_contexes, streams, av_format_context reserved !
    i = avformat_write_header(av_format_context, NULL);
    if (i < 0) {
        logger.log(LogLevel::fatal) << "MuxFrameFilter : initMux : Error occurred while muxing" << std::endl;
        perror("libValkka: MuxFrameFilter: initMux");
        exit(2);
        // av_err2str(i)
        // closeMux();
        // return;
    }
    //*/
    
    /*
    // test re-write // works OK at this point (before writing any actual frames)
    AVRational fps = AVRational();
    fps.num = 10;
    fps.den = 1;
    streams[0]->avg_frame_rate = fps;
    i=avformat_write_header(av_format_context, NULL); // re-write
    */
    
    // so far so good ..
    if (zerotime > 0) { // user wants to set time reference explicitly and not from first arrived packet ..
        mstimestamp0 = zerotime;
        zerotimeset = true;
    }
    else {
        zerotimeset  =false;
    }

}


void MuxFrameFilter::closeMux() {
    int i;
    
    if (initialized) {
        // std::cout << "MuxFrameFilter: closeMux" << std::endl;
        // avio_closep(&avio_ctx);        
        avformat_free_context(av_format_context);
        
        for (auto it = codec_contexes.begin(); it != codec_contexes.end(); ++it) {
            if (*it != NULL) {
                // std::cout << "MuxFrameFilter: closeMux: context " << (long unsigned)(*it) << std::endl;
                avcodec_close(*it);
                avcodec_free_context(&*it);
                *it = NULL;
            }
        }
        for (auto it = streams.begin(); it != streams.end(); ++it) {
            if (*it != NULL) {
                // std::cout << "MuxFrameFilter: closeMux: stream" << std::endl;
                // eh.. nothing needed here.. enough to call close on the context
                *it = NULL;
            }
        }
    }
    initialized = false;
}


void MuxFrameFilter::deActivate_() {
    if (initialized) {
        av_write_trailer(av_format_context);
        closeMux();
    }
    active=false;
}


void MuxFrameFilter::go(Frame* frame) {
    std::unique_lock<std::mutex> lk(this->mutex);
    
    // std::cout << "MuxFrameFilter: go: frame " << *frame << std::endl;
    
    // make a copy of the setup frames ..
    if (frame->getFrameClass() == FrameClass::setup) { // SETUPFRAME
        SetupFrame *setupframe = static_cast<SetupFrame*>(frame);        
        if (setupframe->sub_type == SetupFrameType::stream_init) { // INIT
            if (setupframe->subsession_index>1) {
                logger.log(LogLevel::fatal) << "MuxFrameFilter : too many subsessions! " << std::endl;
            }
            else {
                logger.log(LogLevel::debug) << "MuxFrameFilter :  go : got setup frame " << *setupframe << std::endl;
                setupframes[setupframe->subsession_index].copyFrom(setupframe);
            }
            return;
        } // INIT
    } // SETUPFRAME
    
    else if (frame->getFrameClass() == FrameClass::basic) {
        BasicFrame *basicframe = static_cast<BasicFrame*>(frame);
        
        if (!ready) {
            if (setupframes[0].subsession_index > -1) { // we have got at least one setupframe and after that, payload
                ready=true;
            }
        }
        
        if (ready and active and !initialized) { // got setup frames, writing has been requested, but file has not been opened yet
            initMux(); // modifies member initialized
            if (!initialized) { // can't init this file.. de-activate
                deActivate_();
            }
        }
        
        if (initialized) { // everything's ok! just write..
            if (!zerotimeset) {
                mstimestamp0 = basicframe->mstimestamp;
                zerotimeset = true;
            }
            long int dt = (basicframe->mstimestamp-mstimestamp0);
            if (dt < 0) { dt = 0; }
            
            logger.log(LogLevel::debug) << "MuxFrameFilter : writing frame with mstimestamp " << dt << std::endl;

            /*
            // this copy is unnecessary:
            internal_frame.copyFrom(basicframe);
            internal_frame.mstimestamp = dt;
            internal_frame.fillAVPacket(avpkt);
            */            
            basicframe->fillAVPacket(avpkt); // copies metadata to avpkt, points to basicframe's payload
            
            if (dt >= 0) {
                avpkt->pts=(int64_t)dt;
            }
            else {
                avpkt->pts=AV_NOPTS_VALUE;
            }
            
            av_interleaved_write_frame(av_format_context, avpkt); // => this calls write_packet // used to crasssshh here, but not anymore, after we have defined dummy read & seek functions!
            // av_write_frame(av_format_context, avpkt);
        }
        else {
            // std::cout << "MuxFrameFilter: go: discarding frame" << std::endl;
        }
    } // BasicFrame
    else { // don't know how to handle that frame ..
    }
}


int MuxFrameFilter::write_packet(void *opaque, uint8_t *buf, int buf_size) {
    // what's coming here?  A complete muxed "frame" or a bytebuffer with several frames.  The frames may continue in the next bytebuffer.
    // Once tje frag_size has been set to a small value, this starts getting complete frames, instead of several frames in the same bytebuffer
    
    MuxFrameFilter* me = static_cast<MuxFrameFilter*>(opaque);
    BasicFrame& internal_frame = me->internal_frame;
    uint32_t &missing = me->missing;
    uint32_t &ccf = me->ccf;
    
    /*
     
    ffmpeg buffers:
    
    0         cc       cc+len   buf_size-1
    |------|---|---------|--|-----|      |----|---|--------|--|----|
    
    frame payload:
    
    new round with len & missing
                 
                  cff
    0             | len
    .......+++++++......
    |------------------------|
                  ............
                    missing
    */
    
    uint32_t cc = 0;
    uint32_t len = 0;
    int i;
    
    
    #ifdef MUXPARSE
    std::cout << "\nbuf_size: " << buf_size << std::endl;
    std::cout << "dump: ";
    #endif
    for(i=0; i <= 6; i++) {
        std::cout << int(buf[i]) << " ";
    }
    std::cout << std::endl;
    
    while (cc < buf_size) {
        if (missing > 0) {
            #ifdef MUXPARSE
            std::cout << "taking up missing bytes: " << missing << std::endl;
            #endif
            len = missing;
        }
        else {
            #ifdef MUXPARSE
            std::cout << "start: ";
            #endif
            for(i=0; i <= 6; i++) {
                std::cout << int(buf[cc+i]) << " ";
            }
            len = deserialize_uint32_big_endian(buf+cc); // resolve the packet length from the mp4 headers
            
            if (len > 99999999) { // absurd value .. this bytestream parser has gone sour.
                std::cout << "MuxFrameFilter: overflow: len: " << len << std::endl;
                exit(2);
            }
            else if (len < 1) {
                logger.log(LogLevel::fatal) << "MuxFrameFilter: packet of length zero!" << std::endl;
                cc += 4;
                continue;
            }
            #ifdef MUXPARSE
            std::cout << "/ len: " << len << std::endl;
            #endif
            
            internal_frame.reserve(len); // does nothing if already has this capacity
            internal_frame.resize(len);
        }
        
        #ifdef MUXPARSE
        std::cout << "cc + len: " << cc + len << std::endl;
        #endif
        
        if ((cc + len) > buf_size) { // required number of bytes is larger than the buffer
            // are there missing bytes from the previous round?
            memcpy(internal_frame.payload.data() + ccf, buf, buf_size - cc); // copy the rest of the buffer
            ccf += buf_size - cc;
            missing = len - (buf_size - cc); // next time this is called, ingest more bytes
            #ifdef MUXPARSE
            std::cout << "missing bytes: " << missing << std::endl;
            #endif
        }
        else { // all required bytes are in the buffer
            memcpy(internal_frame.payload.data() + ccf, buf + cc, len);
            missing = 0;
            #ifdef MUXPARSE
            std::cout << "MuxFrameFilter: OUT: len: " << internal_frame.payload.size() << " dump:" << internal_frame.dumpPayload() << std::endl;
            #endif
            ccf = 0;
            
            // TODO: internal_frame.inspect() => fills container format specific parameters
            
            me->run(&internal_frame);
        }
        cc += len;
    }
}
// 

void MuxFrameFilter::activate(long int zerotime) {
  std::unique_lock<std::mutex> lk(this->mutex);
  if (active) {
    deActivate_();
  }
  
  this->zerotime  =zerotime;
  this->active    =true;
}  


void MuxFrameFilter::deActivate() {
  std::unique_lock<std::mutex> lk(this->mutex);
  
  // std::cout << "FileFrameFilter: deActivate:" << std::endl;
  deActivate_();
  // std::cout << "FileFrameFilter: deActivate: bye" << std::endl;
}
  



FragMP4MuxFrameFilter::FragMP4MuxFrameFilter(const char* name, FrameFilter *next) : MuxFrameFilter(name, next) {
}
    
FragMP4MuxFrameFilter::~FragMP4MuxFrameFilter() {
}

void FragMP4MuxFrameFilter::defineMux() {
    format_name = std::string("mp4");

    // -movflags empty_moov+omit_tfhd_offset+frag_keyframe+separate_moof -frag_size
    av_dict_set(&av_dict, "movflags", "empty_moov+omit_tfhd_offset+frag_keyframe+separate_moof", 0);

    // av_dict_set(&av_dict, "frag_size", "500", 500); // nopes
    av_dict_set(&av_dict, "frag_size", "512", 0);
}
  
  
