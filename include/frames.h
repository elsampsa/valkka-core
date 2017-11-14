#ifndef FRAMES_HEADER_GUARD 
#define FRAMES_HEADER_GUARD
/*
 * frames.h : Valkka Frame class declaration, base class FrameFilter for frame filters
 * 
 * Copyright 2017 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of Valkka library. 
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Valkka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Valkka.  If not, see <http://www.gnu.org/licenses/>. 
 * 
 */

/** 
 *  @file    frames.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1.0 
 *  
 *  @brief Frame definition, base class for FrameFilters
 *
 */


#include "common.h"
#include "sizes.h"
#include "avdep.h"
#include "opengl.h"

/** Various H264 frame types
 * 
 */
namespace H264SliceType {
  static const unsigned none  =0;
  static const unsigned sps   =7;
  static const unsigned pps   =8;
  static const unsigned i     =5;
  static const unsigned pb    =1;
}


/** Different frame types recognized by Frame
 * 
 */
enum class FrameType {
  none,     ///< Uninitialized frame
  avframe,  ///< All relevant data is behind the Frame::av_frame and Frame::av_codec_context internal pointers
  yuvframe, ///< All relevant data is behind the Frame::yuvpbo internal pointer
  setup,    ///< This frame contains data obtained from an rtsp setup / sdp file in Frame::pcmu_pars
  h264,     ///< H264 slice. Payload in Frame::payload, additional data in Frame::h264_pars
  pcmu      ///< PCM-ulaw. Payload in Frame::payload, additional data in Frame::pcmu_pars
};

struct SetupPars { ///< Information corresponding to FrameType::setup
  // AVCodecID codec_id; //https://ffmpeg.org/doxygen/3.0/group__lavc__core.html#gaadca229ad2c20e060a14fec08a5cc7ce
  FrameType      frametype;
  // unsigned short subsession_index;
};
std::ostream &operator<<(std::ostream &os, SetupPars const &m);

struct AVFramePars { ///< Information corresponding to FrameType::avframe
};
std::ostream &operator<<(std::ostream &os, AVFramePars const &m);

struct YUVFramePars { ///< Information corresponding to FrameType::yuvframe
  BitmapType bmtype;
  AVPixelFormat pix_fmt;
};
std::ostream &operator<<(std::ostream &os, YUVFramePars const &m);

// A SWIG note here: https://stackoverflow.com/questions/5508182/static-const-int-causes-linking-error-undefined-reference
struct H264Pars { ///< Information corresponding to FrameType::h264
  short unsigned slice_type;
};
std::ostream &operator<<(std::ostream &os, H264Pars const &m);

struct PCMUPars { ///< Information corresponding to FrameType::pcmu
  // static const AVCodecID codec_id;
};
std::ostream &operator<<(std::ostream &os, PCMUPars const &m);



/** A universal frame class encapsulating all media formats.
  *  
  * So, why not define a base frame class and then subclassing for each media type, i.e. Frame => H264Frame, PCMUFrame, .. and for decoded: YUV420Frame, etc.  where H264Frame and YUV420Frame would have different methods, say
  * H264Frame for peeking the frame type (intra frame, sps, pps, etc.), and YUV420Frame methods for extracting luma, chroma, etc.
  * 
  * In the multithreading fifos we want to reserve the frame objects beforehand.  This way we avoid constant memory (de)allocations.  However, not knowing what frames to expect, we can only instantiate the base class Frame.
  * 
  * This class is a bit like ffmpeg's AVPacket structure.  Another possibility would be to use AVPackets only, but that's not flexible enough for our needs.
  * 
  * Creating a deep copy of a Frame instance (including a copy of the payload):
  * 
  * \code
  * Frame *f1;
  * Frame *f2
  * 
  * f1=new Frame();
  * f2=new Frame();
  * 
  * *(f1) = *(f2)
  * \endcode
  * 
  * I.e. it can be done directly with the "=" operator.  The underlying vector smart pointer takes care of payload copying and resizing the payload in the target frame.
  * 
  * If the target frame is a frame owned by a fifo queue, then the payload in that frame will also grow in order to fit the payload.  This way fifos will adapt automagically to the needed payload size.
  * 
  * Payload data is found for encoded data typically from the Frame::payload bytebuffer.  Depending on Frame::frametype, it can also be found from Frame::av_frame or Frame::yuvpbo.  Frame::av_frame and Frame::yuvpbo are not managed (allocated/freed) from Frame.
  * 
  * 
  * @ingroup frames_tag
  * 
  */  
class Frame {
 
public:
  Frame();          ///< Default constructor
  virtual ~Frame(); ///< Virtual destructor
  
public:
  // void setNTPTimestamp(struct timeval t);  ///< Set PTS timestamp in NTP format // NOT USED
  // void calcRelMsTimestamp (struct timeval t); ///< Calculate relative millisecond timestamp // NOT USED
  void reserve(std::size_t n_bytes);          ///< Reserve space for internal payload
  void resize(std::size_t n_bytes);           ///< Init space for internal payload
  void reset();                               ///< Resets frame to FrameType::none, nulls Frame::av_frame, Frame::av_codec_context and Frame::yuvpbo
  std::string dumpPayload();                  ///< returns std::string with beginning of the payload
  std::string dumpAVFrame();                  ///< returns std::string with info about the ffmpeg AVFrame structure
  void fillPars();                            ///< Inspects payload and fills frame parameters (i.e. H264Pars, etc.) 
  
public: // getters
  long int getMsTimestamp();         ///< Returns the PTS in unix epoch milliseconds
  // long int getRelMsTimestamp();               ///< Return the relative millisecond timestamp // NOT USED
  // unsigned int getSlot(); // TODO
  
public: // setters
  void setMsTimestamp (long int t);  ///< Set PTS in unix epoch millisecond timestamp
  
public: // copying
  void copyMeta(Frame* f);                    ///< Copy public metadata across different frame types (timestamp, subsession_index, pars) to f
    
public: // public metadata
  FrameType frametype;    ///< Type of the frame
  // Frame structures are pre-allocated, so we drag along all (small) auxiliary data types
  SetupPars  setup_pars = {FrameType::none};      ///< Used, if frametype == FrameTypes::setup
  H264Pars   h264_pars  = {H264SliceType::none};  ///< Used, if frametype == FrameTypes::h264
  PCMUPars   pcmu_pars  = {};                     ///< Used, if frametype == FrameTypes::pcmu 
  AVFramePars av_pars   = {};                     ///< Used, if frametype == FrameTypes::avframe
  YUVFramePars yuv_pars = {BitmapPars::notype, AV_PIX_FMT_NONE};    ///< Used, if frametype == FrameTypes::yuvframe
    
public: // public metadata you might want to copy between different frametypes
  long int mstimestamp;                  ///< Presentation time stamp (PTS) in milliseconds
  // long int          rel_mstimestamp;              ///< Relative millisecond timestamp : relative timestamp = (abs mstimestamp - current mstime) // NOT USED!
  unsigned short    subsession_index;             ///< Media subsession index
  SlotNumber        n_slot;                       ///< Slot number identifying the media source
  
public: // payloads, _the_ data (and some FFmpeg metadata as well)
  std::vector<uint8_t> payload;  ///< Raw payload data (use .data() to get the pointer from std::vector) 
  //
  // When using vectors in the dequeues:
  // .. when we issue, for two different frames
  // f1=f2
  // then the copy constructor of Frame f2 is called
  // that copy constructor will call the copy constructor of all internal variables of this class
  // .. and copy constructor of std::vector makes a copy of the whole data .. which we don't want (unless explicitly required)
  //
  // another possibility would be to use in the deque, a deque of pointers .. a possibility which we will use
  /*
  uint8_t* payload;
  uint     payloadmaxlen; ///< max. space reserved for payload
  uint     payloadlen;    ///< space actually used by the payload
  */
  AVFrame* av_frame;                ///< This is a pointer to short-lived, temporary storage, used by the FFmpeg decoder (and finally overwritten by it).  The idea is, that the data is copied from here asap, to *_pars structure and to payload.  Just a pointer.  NOT MANAGED by Frame.  Managed by a Decoder instance,
  AVCodecContext* av_codec_context; ///< NOT MANAGED by Frame. Managed by a Decoder instance.
  YUVPBO* yuvpbo;                   ///< NOT MANAGED by Frame. Managed by an OpenGLThread instance. 
  
public: // operator defs here
  friend std::ostream &operator<<(std::ostream &os, Frame const &m) {
    os << "<Frame: size="<<m.payload.size()<<"/"<<m.payload.capacity()<<" timestamp="<<m.mstimestamp<<" subsession_index="<<m.subsession_index<<" slot="<<m.n_slot<<" / ";  //<<std::endl;
    if (m.frametype==FrameType::none)       { os << "NULL"; }
    else if (m.frametype==FrameType::setup) { os << m.setup_pars; }
    else if (m.frametype==FrameType::h264)  { os << m.h264_pars; }
    else if (m.frametype==FrameType::pcmu)  { os << m.pcmu_pars; }
    else if (m.frametype==FrameType::avframe)   { os << "AVFRAME"; }
    else if (m.frametype==FrameType::yuvframe)  { os << "YUVPBO: " << *(m.yuvpbo); }
    os << ">";
    return os;
  }
  
};
  

/** The mother class of all frame filters!  
  * FrameFilters are used to create "filter chains".  These chains can be used to manipulate the frames, feed them to fifo's, copy them, etc.
  * 
  * @ingroup filters_tag
  */
class FrameFilter { // <pyapi>
  
public: // <pyapi>
  /** Default constructor
   * 
   * @param name  Name of the filter
   * @param next  Next FrameFilter instance in the filter chain
   * 
   */
  FrameFilter(const char* name, FrameFilter* next=NULL); // don't include into the python api (this class is abstract)
  virtual ~FrameFilter();                                ///< Virtual destructor // <pyapi>
  
protected:
  std::string  name;
  FrameFilter* next; ///< The next frame filter in the chain to be applied
  
protected: // <pyapi>
  // does the filtering 
  virtual void go(Frame* frame) = 0; ///< Does the actual filtering/modification to the Frame.  Define in subclass
  
public: // API
  /** Calls this->go(Frame* frame) and then calls the this->next->run(Frame* frame) (if this->next != NULL)
   */
  virtual void run(Frame* frame);
}; // <pyapi>


/** A "hello world" demo class: prints its own name if verbose is set to true.
 * @ingroup filters_tag
 */
class DummyFrameFilter : public FrameFilter { // <pyapi>
  
public: // <pyapi>
  DummyFrameFilter(const char* name, bool verbose=true, FrameFilter* next=NULL); ///< @copydoc FrameFilter::FrameFilter // <pyapi>
  
protected:
  bool verbose;
  
protected:
  void go(Frame* frame);
  
}; // <pyapi>


#endif
