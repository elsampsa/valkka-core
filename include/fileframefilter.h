#ifndef fileframefilter_HEADER_GUARD 
#define fileframefilter_HEADER_GUARD

/*
 * fileframefilter.h : File input to matroska files
 * 
 * Copyright 2017-2023 Valkka Security Ltd. and Sampsa Riikonen
 * Copyright 2024 Sampsa Riikonen
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
 *  @file    file.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.6.1 
 *  
 *  @brief   File input to matroska files
 */ 


/*

Some ideas:

  * SetupStorageFrameFilter (stores any setup frames it observes)
  
Writing streams:

  * By a FrameFilter
  * By a Thread that uses that FrameFilter .. a service, similar to LiveThread or OpenGLThread
  
Reading streams:

  * A service thread, similar to LiveThread ..?  
  * .. could do both writing and reading
  
Extend LiveThread into sending frames over rtsp (or sdp)


*/

#include "constant.h"
#include "framefilter.h"


/** Add state information to stream
 * 
 * - Add SetupFrames to a stream
 * - If codec change is detected, then send a new set of SetupFrames
 * - This is only for one slot (i.e. not for multiple streams)
 * 
 */
class InitStreamFrameFilter : public FrameFilter {                          // <pyapi>
    
public:                                                                     // <pyapi>
    InitStreamFrameFilter(const char* name, FrameFilter *next = NULL);      // <pyapi>
    ~InitStreamFrameFilter();                                               // <pyapi>
    
protected:
    std::vector<SetupFrame> setupframes;  ///< cached setupframes

protected:
    void go(Frame* frame);
    void run(Frame* frame);
};                                                                          // <pyapi>



/** Pipe stream into a matroska (mkv) file
 * 
 * This FrameFilter should be connected to the live stream at all times: it observes the setup frames (FrameTypes::setup) and saves them internally.
 * 
 * When the FileFrameFilter::activate method is called, it configures the files accordingly, and starts streaming into disk.
 * 
 * Notes about FFmpeg muxing & file output:
 * 
 * 
 * AVCodecContext  : defines the codec
 * AVFormatContext : defines the (de)muxing & has pointers to AVIOContext (in the ->pb member)
 * AVIOContext     : defines the input / output file
 * AVStream        : raw payload (video track, audio track)
 * 
 * 
 * 
 * avformat_alloc_output_context2(AVFormatContext* av_format_context, NULL, "matroska", NULL);
 * 
 * avio_open(AVIOContext* av_format_context->pb, filename.c_str(), AVIO_FLAG_WRITE);
 * 
 * AVStream* av_stream = avformat_new_stream(AVFormatContext* av_format_context, AVCodec* av_codec_context->codec);
 * 
 * Actual writing like this:
 * 
 * av_interleaved_write_frame(AVFormatContext* av_format_context, avpkt);
 * 
 * For an actual muxer implementation, see for example libavformat/movenc.c : ff_mov_write_packet
 * => avio_write(pb, reformatted_data, size);  (i.e. it uses the AVIOContext) ==> https://ffmpeg.org/doxygen/2.8/aviobuf_8c_source.html#l00178
 * 
 * AVIOContext is C-style "class" that can be re-defined: https://ffmpeg.org/doxygen/2.8/structAVIOContext.html
 * 
 * A custom AVIOContext can be created with "avio_alloc_context"
 * 
 * 
 * 
 * @ingroup filters_tag
 * @ingroup file_tag
 */
class FileFrameFilter : public FrameFilter {                              // <pyapi>
  
public:                                                                   // <pyapi>  
    /** Default constructor 
    *
    * @param name    name
    * @param next    next framefilter to be applied in the filterchain
    */
    FileFrameFilter(const char *name, FrameFilter *next=NULL);              // <pyapi>
    /** Default destructor */
    ~FileFrameFilter();                                                     // <pyapi>

protected:
    bool active;                       ///< Writing to file has been requested (but not necessarily achieved..)
    bool ready;                        ///< Got enough setup frames
    bool initialized;                  ///< File was opened ok : codec_contexes, streams and av_format_context reserved (should be freed at some point)
    long int mstimestamp0;             ///< Time of activation (i.e. when the recording started)
    long int zerotime;                 ///< Start time set explicitly by the user
    bool zerotimeset;
    std::string filename;
  
protected: //libav stuff
    AVRational timebase;
    std::vector<AVCodecContext*>  codec_contexes;
    std::vector<AVStream*>        streams;
    AVFormatContext               *av_format_context;
    AVPacket                      *avpkt;
  
protected: //mutex stuff
    std::mutex              mutex;     ///< Mutex protecting the "active" boolean
    std::condition_variable condition; ///< Condition variable for the mutex

protected: //frames
    std::vector<SetupFrame>       setupframes;        ///< deep copies of the arrived setup frames
    BasicFrame                    internal_frame;     ///< copy of the arrived frame and payload
  
protected:
    void go(Frame* frame);
    void initFile();           ///< Open file, reserve codec_contexes, streams, write preamble, set initialized=true if success
    void closeFile();          ///< Close file, dealloc codec_contexes, streams
    void deActivate_();
    void writeHeader();
  
public: // API calls                                                                         // <pyapi>
    // setFileName(const char* fname); ///< Sets the output filename                           // <pyapi>
    void activate(const char* fname, long int zerotime=0);       ///< Request streaming to disk asap (when config frames have arrived) // <pyapi>
    void deActivate();                                           ///< Stop streaming to disk   // <pyapi>
};                                                                                           // <pyapi>


#endif
