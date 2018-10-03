#ifndef fileframefilter_HEADER_GUARD 
#define fileframefilter_HEADER_GUARD

/*
 * fileframefilter.h : File input to matroska files
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
 *  @file    file.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.7.0 
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

/** Pipe stream into a matroska (mkv) file
 * 
 * This FrameFilter should be connected to the live stream at all times: it observes the setup frames (FrameTypes::setup) and saves them internally.
 * 
 * When the FileFrameFilter::activate method is called, it configures the files accordingly, and starts streaming into disk.
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
  bool initialized;                  ///< File was opened ok : contexes, streams and output_context reserved (should be freed at some point)
  long int mstimestamp0;             ///< Time of activation (i.e. when the recording started)
  long int zerotime;                 ///< Start time set explicitly by the user
  bool zerotimeset;
  std::string filename;
  
protected: //libav stuff
  AVRational timebase;
  std::vector<AVCodecContext*>  contexes;
  std::vector<AVStream*>        streams;
  AVFormatContext               *output_context;
  AVPacket                      *avpkt;
  
protected: //mutex stuff
  std::mutex              mutex;     ///< Mutex protecting the "active" boolean
  std::condition_variable condition; ///< Condition variable for the mutex
  
protected: //frames
  std::vector<SetupFrame>       setupframes;        ///< deep copies of the arrived setup frames
  BasicFrame                    internal_frame;     ///< copy of the arrived frame and payload
  
protected:
  void go(Frame* frame);
  void initFile();           ///< Open file, reserve contexes, streams, write preamble, set initialized=true if success
  void closeFile();          ///< Close file, dealloc contexes, streams
  void deActivate_();
  
public: // API calls                                                                         // <pyapi>
  // setFileName(const char* fname); ///< Sets the output filename                           // <pyapi>
  void activate(const char* fname, long int zerotime=0);       ///< Request streaming to disk asap (when config frames have arrived) // <pyapi>
  void deActivate();                                           ///< Stop streaming to disk   // <pyapi>
};                                                                                           // <pyapi>


#endif
