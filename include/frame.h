#ifndef frame_HEADER_GUARD
#define frame_HEADER_GUARD

/*
 * frame.h : Frame classes
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
 *  @file    frame.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.0.3 
 *  
 *  @brief   Frame classes
 */

#include "common.h"
#include "codec.h"
#include "metadata.h"
#include "threadsignal.h"
#include "constant.h"
#include "avdep.h"
#include "opengl.h"
#include "tools.h"
#include "macro.h"
#include "rawrite.h"

/** Enumeration of Frame classes used by Valkka
 * 
 * @ingroup frames_tag
 */
enum class FrameClass
{
    none, ///< unknown

    basic, ///< data at payload

    avpkt, ///< data at ffmpeg avpkt

    avmedia,  ///< data at ffmpeg av_frame and ffmpeg av_codec_context
    avbitmap, ///< child of avmedia: video
    // avbitmap_np,  ///< child of avmedia: video, non-planar

    avaudio, ///< child of avmedia: audio

    avrgb, ///< rgb interpolated from yuv

    yuv, ///< data at the GPU

    rgb, ///< our own RGB24 data structure

    setup,  ///< setup data
    signal, ///< signal to AVThread or OpenGLThread.  Also custom signals to custom Threads

    marker, ///< Used when sending blocks of frames: mark filesystem and block start and end

    mux, ///< Muxed streams, for example, MP4 or matroska

    First = none,
    Last = mux
};

/** Methods to correct frame timestamps
 * 
 */
enum class TimeCorrectionType // <pyapi>
{          // <pyapi>
    none,  // <pyapi>
    smart, // <pyapi>
    dummy  // <pyapi>
};         // <pyapi>

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
class Frame
{

public:
    Frame();          ///< Default ctor
    virtual ~Frame(); ///< Default virtual dtor
    frame_essentials(FrameClass::none, Frame);
    frame_clone(FrameClass::none, Frame);
    /*Frame(const Frame &f); ///< Default copy ctor
  
  
public: // frame essentials : must be defined for each frame subclass
  virtual FrameClass getFrameClass(); ///< Returns the subclass frame type.  See Frame::frameclass
  virtual void copyFrom(Frame *f);    ///< Copies data to this frame from a frame of the same type (also metadata)
  */

public:                                                  // redefined virtual
    virtual void print(std::ostream &os) const;          ///< Produces frame output
    virtual std::string dumpPayload();                   ///< Dumps internal payload data
    virtual void dumpPayloadToFile(std::ofstream &fout); ///< Dumps internal payload data into a file
    virtual void updateAux();                        ///< Update internal auxiliary state variables
    virtual void update();                           ///< Update helper points (call always)
    virtual void reset();                                ///< Reset the internal data
    virtual bool isSeekable();                           ///< Can we seek to this frame? (e.g. is it a key-frame .. for H264 sps packets are used as seek markers)

public:
    void copyMetaFrom(Frame *f); ///< Copy metadata (slot, subsession index, timestamp) to this frame

protected:
    FrameClass frameclass; ///< Declares frametype for correct typecast.  Used by Frame::getFrameClass()

public:                   // public metadata
    SlotNumber n_slot;    ///< Slot number identifying the media source
    int subsession_index; ///< Media subsession index
    long int mstimestamp; ///< Presentation time stamp (PTS) in milliseconds
};

inline std::ostream &operator<<(std::ostream &os, const Frame &f)
{
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
 * TODO: update to the metadata scheme using metadata.h
 * 
 * @ingroup frames_tag
 */
class BasicFrame : public Frame
{

public:
    BasicFrame();          ///< Default ctor
    virtual ~BasicFrame(); ///< Default virtual dtor
    frame_essentials(FrameClass::basic, BasicFrame);
    frame_clone(FrameClass::basic, BasicFrame);
    /*BasicFrame(const BasicFrame &f); ///< Default copy ctor
  
public: // frame essentials
  virtual FrameClass getFrameClass();         ///< Returns the subclass frame type.  See Frame::frameclass
  virtual void copyFrom(Frame *f);            ///< Copies data to this frame from a frame of the same type
  */

public:                                         // redefined virtual
    virtual void print(std::ostream &os) const; ///< How to print this frame to output stream
    virtual std::string dumpPayload();
    virtual void dumpPayloadToFile(std::ofstream &fout);
    virtual void reset();      ///< Reset the internal data
    virtual bool isSeekable(); ///< for H264 true if sps, other codecs, always true

public:                                // payload handling
    void reserve(std::size_t n_bytes); ///< Reserve space for internal payload
    void resize(std::size_t n_bytes);  ///< Init space for internal payload

public:                           // frame variables
    std::vector<uint8_t> payload; ///< Raw payload data (use .data() to get the pointer from std::vector)
    AVMediaType media_type;       ///< Type of the media (video/audio)
    AVCodecID codec_id;           ///< AVCodeCID of the media

public:                 // codec-dependent parameters
    H264Pars h264_pars; ///< H264 parameters, extracted from the payload

public:                  // codec-dependent functions
    void fillPars();     ///< Fill codec-dependent parameters based on the payload
    void fillH264Pars(); ///< Inspects payload and fills BasicFrame::h264_pars;

public:
    void fillAVPacket(AVPacket *avpkt);                                                                    ///< Copy payload to AVPacket structure
    void copyFromAVPacket(AVPacket *avpkt);                                                                ///< Copy data from AVPacket structure
    void filterFromAVPacket(AVPacket *avpkt, AVCodecContext *codec_ctx, AVBitStreamFilterContext *filter); ///< Copy data from AVPacket structure

public:                                                  // frame serialization
    std::size_t calcSize();                              ///< How much this frame occupies in bytes when serialized
    bool dump(IdNumber device_id, RaWriter &raw_writer); ///< Write the frame to filestream with a certain device id
    IdNumber read(RawReader &raw_reader);                ///< Read the frame from filestream.  Returns device id
};

/** A muxed packet (in some container format)
 * 
 * TODO: isSeekable / Meta / Init:
 * peek into payload ..
 * .. or these are set at the ctor? (discovered by the ffmpex muxer demangling)
 */
class MuxFrame : public Frame {

public:
    MuxFrame(); ///< Default ctor
    virtual ~MuxFrame(); ///< Default virtual dtor
    frame_essentials(FrameClass::mux, MuxFrame);
    frame_clone(FrameClass::mux, MuxFrame);
        
public: // redefined virtual
    virtual void print(std::ostream& os) const;             ///< Produces frame output
    virtual std::string dumpPayload();                      ///< Dumps internal payload data
    virtual void dumpPayloadToFile(std::ofstream& fout);    ///< Dumps internal payload data into a file
    virtual void reset();                                   ///< Reset the internal data
    //virtual bool isSeekable();                              ///< Can we seek to this frame? 

/*
public:
    virtual bool isInit();  ///< for frag-MP4: ftyp, moov
    virtual bool isMeta();  ///< for frag-MP4: moof
    ///< otherwise its payload
*/

public:                                // payload handling
    void reserve(std::size_t n_bytes); ///< Reserve space for internal payload
    void resize(std::size_t n_bytes);  ///< Init space for internal payload

public:
    std::vector<uint8_t> payload; ///< Raw payload data (use .data() to get the pointer from std::vector)
    AVMediaType media_type;       ///< Type of the media (video/audio) of the underlying elementary stream
    AVCodecID codec_id;           ///< AVCodeCID of the underlying elementary stream

public:
    std::vector<uint8_t> meta_blob; ///< Byte blob that is casted to correct metadata struct
    MuxMetaType          meta_type; ///< Mux type that mandates how meta_blob is casted
};


enum class SetupFrameType
{
    none,
    stream_init,
    stream_state
};

/** Setup frame
 * 
 * "Setup Frame" is not maybe the most transparent name.  This frame class carries general information between Threads
 * 
 * - For decoders and muxers signals instantiation and initialization
 * - May carry additional metadata of the stream if necessary (in the future)
 * - Carries information about file stream states (play, stop, seek, etc.)
 * 
 * Copiable/Queable : yes.  uses default copy-constructor and copy-assignment
 * 
 * @ingroup frames_tag
 */
class SetupFrame : public Frame
{

public:
    SetupFrame();          ///< Default ctor
    virtual ~SetupFrame(); ///< Default virtual dtor
    frame_essentials(FrameClass::setup, SetupFrame);
    frame_clone(FrameClass::setup, SetupFrame);
    /*
  SetupFrame(const SetupFrame &f); ///< Default copy ctor
  
public: // frame essentials
  virtual FrameClass getFrameClass();         ///< Returns the subclass frame type.  See Frame::frameclass
  virtual void copyFrom(Frame *f);            ///< Copies data to this frame from a frame of the same type
  */

public:                                         // redefined virtual
    virtual void print(std::ostream &os) const; ///< How to print this frame to output stream
    virtual void reset();                       ///< Reset the internal data

public:                      // managed objects
    SetupFrameType sub_type; ///< Type of the SetupFrame

    AVMediaType media_type; ///< For subtype stream_init
    AVCodecID codec_id;     ///< For subtype stream_init

    AbstractFileState stream_state; ///< For subtype stream_state
};

/** Decoded Frame in FFmpeg format
 * 
 * - The decoded frame is in FFmpeg/libav format in AVMediaFrame::av_frame
 * - Constructor does not reserve data for frames.  This is done by classes using this class
 * 
 * Copiable/Queable : no
 * 
 * @ingroup frames_tag
 */
class AVMediaFrame : public Frame
{

public:
    AVMediaFrame();          ///< Default ctor
    virtual ~AVMediaFrame(); ///< Default virtual dtor
    //frame_essentials(FrameClass::avmedia,AVMediaFrame); // now this is a virtual class ..
    //frame_clone(FrameClass::avmedia,AVMediaFrame);
    /*AVMediaFrame(const AVMediaFrame &f); ///< Default copy ctor
    
    public: // frame essentials
    virtual FrameClass getFrameClass(); ///< Returns the subclass frame type.  See Frame::frameclass
    virtual void copyFrom(Frame *f);    ///< Copies data to this frame from a frame of the same type
    */

/*
public:
    virtual void updateAux() = 0; ///< update any helper objects
    virtual void update() = 0;
*/

public: // redefined virtual
    virtual std::string dumpPayload();
    virtual void print(std::ostream &os) const; ///< How to print this frame to output stream
    virtual void reset();                       ///< Reset the internal data

public:                     // helper objects : values should correspond to member av_frame
    AVMediaType media_type; ///< helper object: media type
    AVCodecID codec_id;     ///< helper object: codec id

public:                // managed objects
    AVFrame *av_frame; ///< The decoded frame
};

/** Decoded YUV/RGB frame in FFMpeg format
 * 
 * - This FrameClass has decoded video, and is used by the VideoDecoder class. AVThread passes this frame down the filterchain.
 * - It's up to the VideoDecoder to call AVBitmapFrame::update() and to update the helper objects (AVBitmapFrame::bmpars, etc.)
 * - Constructor does not reserve data for frames.  This is done by classes using this class, for example by VideoDecoder.  This is desirable as we don't know the bitmap dimensions yet ..
 * 
 * This is an "intermediate frame".  It is typically copied asap into a YUVFrame.
 * 
 * Copiable/Queable : no
 * 
 * @ingroup frames_tag
 */
class AVBitmapFrame : public AVMediaFrame
{

public:
    AVBitmapFrame();          ///< Default ctor
    virtual ~AVBitmapFrame(); ///< Default virtual dtor
    frame_essentials(FrameClass::avbitmap, AVBitmapFrame);
    frame_clone(FrameClass::avbitmap, AVBitmapFrame); // TODO: think about this!
                                                      /*
  AVBitmapFrame(const AVBitmapFrame &f); ///< Default copy ctor
  
public: // frame essentials
  virtual FrameClass getFrameClass(); ///< Returns the subclass frame type.  See Frame::frameclass
  virtual void copyFrom(Frame *f);    ///< Copies data to this frame from a frame of the same type
  */

public: // redefined virtual
    virtual std::string dumpPayload();
    virtual void print(std::ostream &os) const; ///< How to print this frame to output stream
    virtual void reset();                       ///< Reset the internal data
    virtual void copyPayloadFrom(AVBitmapFrame *frame);
    virtual void updateAux();                   ///< Uses AVBitmapFrame::av_frame width and height and AVBitmapFrame::av_pixel_format to calculate AVBitmapFrame::bmpars
    virtual void update();
    virtual void reserve(int width, int height);

public:                            // helper objects
    AVPixelFormat av_pixel_format; ///< From AVCodecContext .. this class implies YUV420P so this is not really required ..
    BitmapPars bmpars;             ///< Calculated bitmap plane dimensions, data sizes, etc.
    uint8_t *y_payload;            ///< shortcut to AVMediaFrame::av_frame->data[0]
    uint8_t *u_payload;            ///< shortcut to AVMediaFrame::av_frame->data[1]
    uint8_t *v_payload;            ///< shortcut to AVMediaFrame::av_frame->data[2]
};

/** Decoded YUV frame in a non-planar format (thus "NP")
 * 
 * For example, the YUYV422 format (AV_PIX_FMT_YUYV422), where the data layout looks like this:
 * YUYV YUYV YUYV YUYV YUYV YUYV
 * 
 * here we could optimize and copy from YUYV422 directly to YUV420 on the GPU
 * like in YUVFrame::fromAVBitmapFrame
 * maybe someday ..
 * 
 
class AVBitmapFrameNP : public AVMediaFrame {
  
public:
  AVBitmapFrameNP(); ///< Default ctor
  virtual ~AVBitmapFrameNP(); ///< Default virtual dtor
  frame_essentials(FrameClass::avbitmap_np, AVBitmapFrameNP);
  frame_clone(FrameClass::avbitmap_np, AVBitmapFrameNP); // TODO: think about this!
  
public: // redefined virtual
  virtual std::string dumpPayload();
  virtual void print(std::ostream& os) const; ///< How to print this frame to output stream
  virtual void reset();              ///< Reset the internal data
  virtual void update();             ///< Uses AVBitmapFrame::av_frame width and height and AVBitmapFrame::av_pixel_format to calculate AVBitmapFrame::bmpars
  
    
public: // helper objects
  AVPixelFormat  av_pixel_format; ///< From AVCodecContext .. this class implies YUV422P so this is not really required ..
  BitmapPars     bmpars;          ///< Reference parameters for corresponding YUV420P frame
  uint8_t*       payload;         ///< shortcut to the non-planar data
};
*/

/** Decoded RGB frame in FFMpeg format
 * 
 * - This FrameClass is produced by SwScaleFrameFilter (that performs YUV=>RGB interpolation).  This FrameClass is used typically when passing frame data to shared memory
 * - Constructor does not reserve data for frames.  This is done by classes using this class (see SwScaleFrameFilter)
 * - Frames of this class can also be pre-reserved after construction, by using the member AVRGBFrame::reserve
 * 
 * Copiable/Queable : yes
 * 
 * @ingroup frames_tag
 * @ingroup shmem_tag
 */
class AVRGBFrame : public AVBitmapFrame
{

public:
    AVRGBFrame();
    virtual ~AVRGBFrame();
    frame_essentials(FrameClass::avrgb, AVRGBFrame);
    frame_clone(FrameClass::avrgb, AVRGBFrame); // TODO: think about this!

public:
    virtual void reserve(int width, int height); ///< Reserve RGB image with width & height

public: // redefined virtual
    virtual std::string dumpPayload();
    virtual void print(std::ostream &os) const; ///< How to print this frame to output stream
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
class YUVFrame : public Frame
{

public:
    YUVFrame(BitmapPars bmpars); ///< Default ctor
    virtual ~YUVFrame();         ///< Default virtual dtor
    frame_essentials(FrameClass::yuv, YUVFrame);
    // TODO: frame_clone
    /*
  YUVFrame(const YUVFrame &f); ///< Default copy ctor
  
public: // frame essentials
  virtual FrameClass getFrameClass(); ///< Returns the subclass frame type.  See Frame::frameclass
  virtual void copyFrom(Frame *f);    ///< Copies data to this frame from a frame of the same type

  */

public:                       // variables filled at constructor time
    BitmapPars bmpars;        // the maximum, pre-reserved size
    BitmapPars source_bmpars; // the actual size of the image

private:
    void reserve(); ///< Reserve data on the GPU.  Used by the constructor only.
    void release(); ///< Releases data on the GPU.  Used by the destructor only.

public:             // variables used/filled by reserve
    GLuint y_index; ///< internal OpenGL/GPU index
    GLuint u_index; ///< internal OpenGL/GPU index
    GLuint v_index; ///< internal OpenGL/GPU index

private:
    GLubyte *y_payload; ///< direct memory access memory address, returned by GPU
    GLubyte *u_payload; ///< direct memory access memory address, returned by GPU
    GLubyte *v_payload; ///< direct memory access memory address, returned by GPU

public: // redefined virtual
    virtual std::string dumpPayload();
    virtual void print(std::ostream &os) const; ///< How to print this frame to output stream
    virtual void reset();

public:
    void fromAVBitmapFrame(AVBitmapFrame *f); ///< Copies data to the GPU from AVBitmapFrame
};

/** Our own RGB24 structure
 * 
 * - Typically, copy the contents of (a temporary) AVRGBFrame here
 * - This frame is used in FrameFifo s
 * 
 */
class RGBFrame : public Frame
{

public:
    RGBFrame(int max_width, int max_height);
    virtual ~RGBFrame();
    frame_essentials(FrameClass::rgb, RGBFrame);

private:
    std::vector<uint8_t> payload; ///< RGB24 data as continuous bytes.  3 bytes per pixel
    int max_width, max_height;
    int width, height;

public: // redefined virtual
    virtual std::string dumpPayload();
    virtual void print(std::ostream &os) const; ///< How to print this frame to output stream
    virtual void reset();

public:
    void fromAVRGBFrame(AVRGBFrame *f); ///< Copies data from (temporary) AVRGBFrame .. set the width & height members
};

typedef std::vector<Frame *> Reservoir;
typedef std::deque<Frame *> Stack;
typedef std::deque<Frame *> Fifo;
typedef std::deque<Frame *> Cache;

typedef std::vector<YUVFrame *> YUVReservoir;
typedef std::deque<YUVFrame *> YUVStack;
typedef std::vector<RGBFrame *> RGBReservoir;
typedef std::deque<RGBFrame *> RGBStack;


/** A frame, signaling internal thread commands, states of recorded video, etc.
 * 
 * libValkka threads read a single fifo.   From this fifo they receive both the media frames (payload)
 * and frames representing API commands given to them.
 * 
 * The "frontend" (API) part inserts a SignalFrame into their fifo, which they then process
 * (most of the libValkka threads have been updated to this technique)
 * 
 * SignalFrames are also used to send additional information between threads and along the downstream pipeline.
 * 
 * This should also be generalizable for extension modules (new threads, custom signals send downstream, etc.)
 * 
 * SignalFrames must be copyable when they are placed into fifos (copy-on-write)
 */
class SignalFrame : public Frame
{

public:
    SignalFrame();          ///< Default ctor
    virtual ~SignalFrame(); ///< Default virtual dtor
    frame_essentials(FrameClass::signal, SignalFrame);
    frame_clone(FrameClass::signal, SignalFrame);

public:
    unsigned signaltype; /// < For making correct typecast of custom_ctx_buf.  See also threadsignal.h

public:
    OpenGLSignalContext opengl_signal_ctx;                 ///< Thread commands to OpenGLThread
    AVSignalContext av_signal_ctx;                         ///< Thread commands to AVThread
    ValkkaFSWriterSignalContext valkkafswriter_signal_ctx; ///< Thread commands to ValkkFSWriterThread
    ValkkaFSReaderSignalContext valkkafsreader_signal_ctx; ///< Thread commands to ValkkaFSReaderThread
    // TODO: those ctxes should be written with the new technique:
    // this generalizes & is copyable
    std::vector<uint8_t> signal_ctx_buf;                ///< A byte-buffer where the signal context is serialized
    /* example of the (revised) signal handling
    if (signalframe->signaltype==SignalType::offline) {
        OfflineSignalContext ctx = OfflineSignalContext();
        get_signal_context(signalframe, ctx);
        # that's a macro: take a look at macro.h
    }
    */

public:
    virtual void reset();

};

class MarkerFrame : public Frame
{

public:
    MarkerFrame();          ///< Default ctor
    virtual ~MarkerFrame(); ///< Default virtual dtor
    frame_essentials(FrameClass::marker, MarkerFrame);
    frame_clone(FrameClass::marker, MarkerFrame);

public: // redefined virtual
    // virtual std::string dumpPayload();
    virtual void print(std::ostream &os) const; ///< How to print this frame to output stream
    virtual void reset();

public:
    bool fs_start, fs_end; ///< Filesystem start / end  // this controlled better at the python level
    bool tm_start, tm_end; ///< Transmission start / end
};

#endif
