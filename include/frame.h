#ifndef frame_HEADER_GUARD
#define frame_HEADER_GUARD

/*
 * frame.h : Frame classes
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
 *  @file    frame.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.5.2 
 *  
 *  @brief   Frame classes
 */ 

#include "common.h"
#include "codec.h"
#include "threadsignal.h"
#include "constant.h"
#include "avdep.h"
#include "opengl.h"
#include "tools.h"

// Macros for making getFrameClass and copyFrom
#define frame_essentials(CLASSNAME, CLASS) \
FrameClass CLASS::getFrameClass() {\
  return CLASSNAME;\
};\
void CLASS::copyFrom(Frame *f) {\
  CLASS *cf;\
  cf=dynamic_cast<CLASS*>(f);\
  if (!cf) {\
    perror("FATAL : invalid cast at copyFrom");\
    exit(5);\
  }\
  *this =*(cf);\
};\



/** Enumeration of Frame classes used by Valkka
 * 
 * @ingroup frames_tag
 */
enum class FrameClass {
  none,      ///< unknown
  
  basic,     ///< data at payload
  
  avpkt,     ///< data at ffmpeg avpkt
  
  avmedia,   ///< data at ffmpeg av_frame and ffmpeg av_codec_context
  avbitmap,  ///< child of avmedia: video
  avaudio,   ///< child of avmedia: audio
  
  avrgb,     ///< rgb interpolated from yuv
  
  yuv,      ///< data at the GPU
  
  setup,     ///< setup data
  signal,    ///< signal to AVThread or OpenGLThread
  
  First=none,
  Last =signal
};



/** Frame: An abstract queueable class.
 * 
 * Instances of this class can be placed into FrameFifos and passed through FrameFilters (see the FrameFifo and FrameFilter classes).  They have characteristic internal data, defined in child classes.
 * 
 * The internal data can be paylod, information for decoder setup, signals to threads, etc.
 * 
 * FrameFifos and FrameFilters are responsible for checking the exact type of the Frame, performing correct typecast and using/discarding/modificating the Frame.
 * 
 * All Frame classes can be put through FrameFilters, but not all Frame classes are copyable/queueable.  Copyable Frame instances can be placed in a FrameFifo (that creates a copy of the Frame before queuing it).
 * 
 * A FrameFifo can also decide to transform the frame to another type before placing it into the queue.
 * 
 * The internal data (and state of the object) consists typically of managed objects, created with external libraries (say, FFmpeg/libav) and "helper" objects/members.  In the case of FFmpeg these are auxiliary members that make the use of the underlying FFmpeg objects more transparent.  State of the managed objects and the helper objects must be kept consistent.
 * 
 * When the state of the managed object(s) is changed, call "updateFrom" with const parameters.  This also makes it more transparent, which parameters trigger updates in helper (and managed) objects.
 * 
 * @ingroup frames_tag
 */
class Frame {
  
public:
  Frame(); ///< Default ctor
  virtual ~Frame(); ///< Default virtual dtor
  
public: // frame essentials : must be defined for each frame subclass
  virtual FrameClass getFrameClass(); ///< Returns the subclass frame type.  See Frame::frameclass
  virtual void copyFrom(Frame *f);    ///< Copies data to this frame from a frame of the same type (also metadata)
  
public: // redefined virtual
  virtual void print(std::ostream& os) const; ///< Produces frame output
  virtual std::string dumpPayload();          ///< Dumps internal payload data
  virtual void dumpPayloadToFile(std::ofstream& fout); ///< Dumps internal payload data into a file
  virtual void update();             ///< Update internal auxiliary state variables
  virtual void reset();              ///< Reset the internal data
  
public:
  void copyMetaFrom(Frame *f);        ///< Copy metadata (slot, subsession index, timestamp) to this frame
  
protected:
  FrameClass frameclass;              ///< Declares frametype for correct typecast.  Used by Frame::getFrameClass()
  
public: // public metadata
  SlotNumber      n_slot;                       ///< Slot number identifying the media source
  int             subsession_index;             ///< Media subsession index
  long int        mstimestamp;                  ///< Presentation time stamp (PTS) in milliseconds  
};

std::ostream& operator<< (std::ostream& os, const Frame& f) {
  // https://stackoverflow.com/questions/4571611/making-operator-virtual
  f.print(os);
  return os;
}


/** Custom payload Frame
 * 
 * Includes codec info and the payload.  Received typically from LiveThread or FileThread.
 * 
 * Copiable/Queueable : yes
 * 
 * @ingroup frames_tag
 */
class BasicFrame : public Frame {

public:
  BasicFrame(); ///< Default ctor
  virtual ~BasicFrame(); ///< Default virtual dtor
  
public: // frame essentials
  virtual FrameClass getFrameClass();         ///< Returns the subclass frame type.  See Frame::frameclass
  virtual void copyFrom(Frame *f);            ///< Copies data to this frame from a frame of the same type
  
public: // redefined virtual
  virtual void print(std::ostream& os) const; ///< How to print this frame to output stream
  virtual std::string dumpPayload();
  virtual void dumpPayloadToFile(std::ofstream& fout);
  virtual void reset();              ///< Reset the internal data
  
public: // payload handling
  void reserve(std::size_t n_bytes);          ///< Reserve space for internal payload
  void resize (std::size_t n_bytes);          ///< Init space for internal payload
  
public: // frame variables
  std::vector<uint8_t> payload;    ///< Raw payload data (use .data() to get the pointer from std::vector)
  AVMediaType   media_type;        ///< Type of the media (video/audio)
  AVCodecID     codec_id;          ///< AVCodeCID of the media

public: // codec-dependent parameters  
  H264Pars      h264_pars;        ///< H264 parameters, extracted from the payload
  
public: // codec-dependent functions
  void fillPars();      ///< Fill codec-dependent parameters based on the payload
  void fillH264Pars();  ///< Inspects payload and fills BasicFrame::h264_pars;

public:
  void fillAVPacket(AVPacket *avpkt);      ///< Copy payload to AVPacket structure
  void copyFromAVPacket(AVPacket *avpkt);  ///< Copy data from AVPacket structure
  void filterFromAVPacket(AVPacket *avpkt, AVCodecContext *codec_ctx, AVBitStreamFilterContext *filter);  ///< Copy data from AVPacket structure
};


/** Setup frame for decoders
 * 
 * Carries information for instantiation and initialization of decoders
 * 
 * Copiable/Queable : yes.  uses default copy-constructor and copy-assignment
 * 
 * @ingroup frames_tag
 */
class SetupFrame : public Frame {
  
public:
  SetupFrame(); ///< Default ctor
  virtual ~SetupFrame(); ///< Default virtual dtor
  
public: // frame essentials
  virtual FrameClass getFrameClass();         ///< Returns the subclass frame type.  See Frame::frameclass
  virtual void copyFrom(Frame *f);            ///< Copies data to this frame from a frame of the same type
  
public: // redefined virtual
  virtual void print(std::ostream& os) const; ///< How to print this frame to output stream
  virtual void reset();              ///< Reset the internal data
  
public: // managed objects
  AVMediaType   media_type;
  AVCodecID     codec_id;
};


/** Decoded Frame in FFmpeg format
 * 
 * The decoded frame is in FFmpeg/libav format in AVMediaFrame::av_frame
 * 
 * Copiable/Queable : no
 * 
 * @ingroup frames_tag
 */
class AVMediaFrame : public Frame {
  
public:
  AVMediaFrame(); ///< Default ctor
  virtual ~AVMediaFrame(); ///< Default virtual dtor
  
public: // frame essentials
  virtual FrameClass getFrameClass(); ///< Returns the subclass frame type.  See Frame::frameclass
  virtual void copyFrom(Frame *f);    ///< Copies data to this frame from a frame of the same type
  
public: // redefined virtual
  virtual std::string dumpPayload();
  virtual void print(std::ostream& os) const; ///< How to print this frame to output stream
  virtual void reset();              ///< Reset the internal data

public: // helper objects
  AVMediaType   media_type;          ///< helper object: media type
  AVCodecID     codec_id;            ///< helper object: codec id
  
public: // managed objects
  AVFrame*       av_frame;          ///< The decoded frame
};


/** Decoded YUV/RGB frame in FFMpeg format
 * 
 * This FrameClass has decoded video, and is used by the VideoDecoder class. AVThread passes this frame down the filterchain.
 * 
 * It's up to the VideoDecoder to call AVBitmapFrame::update() and to update the helper objects (AVBitmapFrame::bmpars, etc.)
 * 
 * This is an "intermediate frame".  It is typically copied asap into a YUVFrame.
 * 
 * Copiable/Queable : no
 * 
 * @ingroup frames_tag
 */
class AVBitmapFrame : public AVMediaFrame {
  
public:
  AVBitmapFrame(); ///< Default ctor
  virtual ~AVBitmapFrame(); ///< Default virtual dtor
  
public: // frame essentials
  virtual FrameClass getFrameClass(); ///< Returns the subclass frame type.  See Frame::frameclass
  virtual void copyFrom(Frame *f);    ///< Copies data to this frame from a frame of the same type
  
public: // redefined virtual
  virtual std::string dumpPayload();
  virtual void print(std::ostream& os) const; ///< How to print this frame to output stream
  virtual void reset();              ///< Reset the internal data
  virtual void update();             ///< Uses AVBitmapFrame::av_frame width and height and AVBitmapFrame::av_pixel_format to calculate AVBitmapFrame::bmpars
  
    
public: // helper objects
  AVPixelFormat  av_pixel_format; ///< From AVCodecContext
  BitmapPars     bmpars;          ///< Calculated bitmap plane dimensions, data sizes, etc.
  uint8_t*       y_payload;       ///< shortcut to AVMediaFrame::av_frame->data[0]
  uint8_t*       u_payload;       ///< shortcut to AVMediaFrame::av_frame->data[1]
  uint8_t*       v_payload;       ///< shortcut to AVMediaFrame::av_frame->data[2]
};

/** Decoded RGB frame in FFMpeg format
 * 
 * This FrameClass is produced by SwScaleFrameFilter (that performs YUV=>RGB interpolation).  This FrameClass is used typically when passing frame data to shared memory
 * 
 * Copiable/Queable : no
 * 
 * @ingroup frames_tag
 * @ingroup shmem_tag
 */
class AVRGBFrame : public AVBitmapFrame {
  
public:
  AVRGBFrame();
  virtual ~AVRGBFrame();
  
public: // frame essentials
  virtual FrameClass getFrameClass(); ///< Returns the subclass frame type.  See Frame::frameclass
  virtual void copyFrom(Frame *f);    ///< Copies data to this frame from a frame of the same type
  
public: // redefined virtual
  virtual std::string dumpPayload();
  virtual void print(std::ostream& os) const; ///< How to print this frame to output stream
};



/*
class AVAudioFrame : public AVMediaFrame {
  
public:
  AVAudioFrame();
  virtual ~AVAudioFrame();
  
public: // frame essentials
  virtual FrameClass getFrameClass(); ///< Returns the subclass frame type.  See Frame::frameclass
  virtual void copyFrom(Frame *f);    ///< Copies data to this frame from a frame of the same type
  virtual std::string dumpPayload();
  virtual void print(std::ostream& os) const; ///< How to print this frame to output stream
  
public:
  virtual void getParametersDecoder(const AVCodecContext *ctx); ///< Extract sample rate
  
public: // ffmpeg media parameters
  AVSampleFormat av_sample_fmt;
};
*/


// Decoder gives AVBitmapFrame => OpenGLThread internal FrameFilter => OpenGLThread FrameFifo::writeCopy(Frame *f) => .. returns once a copy has been made


/** A GPU YUV frame.  The payload is at the GPU.  YUVFrame instances are constructed/destructed by the OpenGLThread.
 * 
 * - Resources on the GPU (target pointers for data upload, OpenGL indexes).  YUVFrame::bmpars describing those resources
 * - YUVFrame::source_bmpars describing the frame dimensions (should be less than equal to YUVFrame::bmpars)
 * - No helper objects need to be updated .. data is imply copied and overwritten when calling YUVFrame::fromAVBitmapFrame
 * 
 * Copiable/Queable : no
 * 
 * Instances of this frame are not copiable when crossing thread borders, however, instances of this frame are used internally by OpenGLThread and OpenGLFrameFifo, and they are being queued and recycled.
 * 
 * For more details, see OpenGLFrameFifo and OpenGLFrameFifo::prepareAVBitmapFrame
 * 
 * @ingroup frames_tag
 * @ingroup openglthread_tag
 */
class YUVFrame : public Frame {
  
public:
  YUVFrame(BitmapPars bmpars); ///< Default ctor
  virtual ~YUVFrame(); ///< Default virtual dtor

public: // variables filled at constructor time
  BitmapPars bmpars;        // the maximum, pre-reserved size
  BitmapPars source_bmpars; // the actual size of the image
  
private:
  void reserve();     ///< Reserve data on the GPU.  Used by the constructor only.
  void release();     ///< Releases data on the GPU.  Used by the destructor only.
  
public: // variables used/filled by reserve
  GLuint   y_index;   ///< internal OpenGL/GPU index
  GLuint   u_index;   ///< internal OpenGL/GPU index
  GLuint   v_index;   ///< internal OpenGL/GPU index
  
private:
  GLubyte* y_payload; ///< direct memory access memory address, returned by GPU
  GLubyte* u_payload; ///< direct memory access memory address, returned by GPU
  GLubyte* v_payload; ///< direct memory access memory address, returned by GPU

public: // frame essentials
  virtual FrameClass getFrameClass(); ///< Returns the subclass frame type.  See Frame::frameclass
  virtual void copyFrom(Frame *f);    ///< Copies data to this frame from a frame of the same type
  
public: // redefined virtual
  virtual std::string dumpPayload();
  virtual void print(std::ostream& os) const; ///< How to print this frame to output stream
  virtual void reset();
  
public:
  void fromAVBitmapFrame(AVBitmapFrame *f);  ///< Copies data to the GPU from AVBitmapFrame
};


typedef std::vector<Frame*>  Reservoir;
typedef std::deque <Frame*>  Stack;
typedef std::deque <Frame*>  Fifo;

typedef std::vector<YUVFrame*>  YUVReservoir;
typedef std::deque <YUVFrame*>  YUVStack;



class SignalFrame : public Frame {

public:
  SignalFrame();           ///< Default ctor
  virtual ~SignalFrame();  ///< Default virtual dtor

public:
  OpenGLSignalContext   opengl_signal_ctx;
  AVSignalContext       av_signal_ctx;
  
public:
  virtual FrameClass getFrameClass(); ///< Returns the subclass frame type.  See Frame::frameclass
  virtual void copyFrom(Frame *f);    ///< Copies data to this frame from a frame of the same type
};


#endif
