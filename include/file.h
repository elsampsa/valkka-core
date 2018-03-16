/*
 * file.h : File input and output to and from matroska files
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
 *  @version 0.3.5 
 *  
 *  @brief   File input and output to and from matroska files
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

#include "filters.h"

/** Pipe stream into a matroska (mkv) file
 * 
 * This FrameFilter should be connected to the live stream at all times: it observes the setup frames (FrameTypes::setup) and saves them internally.
 * 
 * When the FileFrameFilter::activate method is called, it configures the files accordingly, and starts streaming into disk.
 * 
 * @ingroup filters_tag
 * @ingroup file_tag
 */
class FileFrameFilter : public FrameFilter { // <pyapi>
  
public: // <pyapi>  
  /** Default constructor 
   *
   * @param name    name
   * @param next    next framefilter to be applied in the filterchain
   */
  FileFrameFilter(const char *name, FrameFilter *next=NULL);              // <pyapi>
  /** Default destructor */
  ~FileFrameFilter();                                                     // <pyapi>
  
protected:
  bool active;
  bool initialized;
  long int mstimestamp0;             ///< Time of activation (i.e. when the recording started)
  bool zerotimeset;
  std::string filename;
  AVRational timebase;
  
  std::mutex              mutex;     ///< Mutex protecting the "active" boolean
  std::condition_variable condition; ///< Condition variable for the mutex
  
  AVCodecContext                *av_codec_context;  ///< temp variable
  AVStream                      *av_stream;         ///< temp variable
  std::vector<AVCodecContext*>  contexes;
  std::vector<AVStream*>        streams;
  std::vector<Frame*>           setupframes;
  AVFormatContext               *output_context;
  
protected:
  void go(Frame* frame);
  
public: // API calls                                                                         // <pyapi>
  // setFileName(const char* fname); ///< Sets the output filename                           // <pyapi>
  bool activate(const char* fname, long int zerotime=0);       ///< Starts streaming to disk // <pyapi>
  void deActivate();                                           ///< Stop streaming to disk   // <pyapi>
};                                                                                           // <pyapi>


