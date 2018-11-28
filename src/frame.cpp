/*
 * frame.cpp :
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
 *  @file    frame.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.9.0 
 *  
 *  @brief 
 */ 

#include "frame.h"
#include "logging.h"


Frame::Frame() : mstimestamp(0), n_slot(0), subsession_index(-1) {
}

  
Frame::~Frame() {
}
 
 
//frame_essentials(FrameClass::none, Frame);
/*
FrameClass Frame::getFrameClass() {
  return FrameClass::none;
}
 
void Frame::copyFrom(Frame *f) {
  Frame *cf;
#ifdef CAST_CHECK
  cf=dynamic_cast<Frame*>(f);
  if (!cf) {
    perror("FATAL : invalid cast at copyFrom")
    exit(5);
  }
#elsed
  cf=static_cast<Frame*>(f);
#endif
  *this =*(cf);
}
*/
 
 
void Frame::print(std::ostream &os) const {
  os << "<Frame: timestamp="<<mstimestamp<<" subsession_index="<<subsession_index<<" slot="<<n_slot<<">";
}
 

void Frame::copyMetaFrom(Frame *f) {
  this->n_slot          =f->n_slot;
  this->subsession_index=f->subsession_index;
  this->mstimestamp     =f->mstimestamp;
}


std::string Frame::dumpPayload() {
  return std::string("");
}


void Frame::dumpPayloadToFile(std::ofstream& fout) {
}


void Frame::reset() {
    this->n_slot          =0;
    this->subsession_index=-1;
    this->mstimestamp     =0;
}

bool Frame::isSeekable() {
    return True;
}


void Frame::update() {
}

  
BasicFrame::BasicFrame() : Frame(), codec_id(AV_CODEC_ID_NONE), media_type(AVMEDIA_TYPE_UNKNOWN), h264_pars(H264Pars()) {
}


BasicFrame::~BasicFrame() {
}


//frame_essentials(FrameClass::basic, BasicFrame);
/* // that macro expands to..

FrameClass BasicFrame::getFrameClass() {
  return FrameClass::basic;
}


void BasicFrame::copyFrom(Frame *f) {
  BasicFrame *cf;
#ifdef CAST_CHECK
  cf=dynamic_cast<BasicFrame*>(f);
  if (!cf) {
    perror("FATAL : invalid cast at copyFrom")
    exit(5);
  }
#else
  cf=static_cast<BasicFrame*>(f);
#endif
  *this =*(cf);
}
*/


void BasicFrame::print(std::ostream &os) const {
  os << "<BasicFrame: timestamp="<<mstimestamp<<" subsession_index="<<subsession_index<<" slot="<<n_slot<<" / ";  //<<std::endl;
  os << "payload size="<<payload.size()<<" / ";
  if (codec_id==AV_CODEC_ID_H264) {os << h264_pars;}
  else if (codec_id==AV_CODEC_ID_PCM_MULAW) {os << "PCMU: ";}
  os << ">";
}


std::string BasicFrame::dumpPayload() {
  std::stringstream tmp;
  for(std::vector<uint8_t>::iterator it=payload.begin(); it<min(payload.end(),payload.begin()+20); ++it) {
    tmp << int(*(it)) <<" ";
  }
  return tmp.str();
}


void BasicFrame::dumpPayloadToFile(std::ofstream& fout) {
  std::copy(payload.begin(), payload.end(), std::ostreambuf_iterator<char>(fout));
}


void BasicFrame::reset() {
  Frame::reset();
  codec_id   =AV_CODEC_ID_NONE;
  media_type =AVMEDIA_TYPE_UNKNOWN;
}


bool BasicFrame::isSeekable() {
    fillPars();
    
    /* // nopes
    if (force_seekable) { // this only for filesystem debuggin
        return true;
    }
    */
    
    switch(codec_id) {
        case AV_CODEC_ID_H264:
            if (h264_pars.slice_type == H264SliceType::sps) {
                return True;
            }
            else {
                return False;
            }
            break;
        default:
            return True;
            break;
    }
}

/*
void BasicFrame::setSeekable() {
    force_seekable = true;
}

void BasicFrame::unSetSeekable() {
    force_seekable = false;
}
*/


void BasicFrame::reserve(std::size_t n_bytes) {
  this->payload.reserve(n_bytes);
}


void BasicFrame::resize(std::size_t n_bytes) {
  this->payload.resize(n_bytes,0);
}


void BasicFrame::fillPars() {
  if (codec_id==AV_CODEC_ID_H264) {
    fillH264Pars();
  }
}


void BasicFrame::fillH264Pars() {
  if (payload.size()>(nalstamp.size()+1)) { 
    h264_pars.slice_type = ( payload[nalstamp.size()] & 31 );
  }
}


void BasicFrame::fillAVPacket(AVPacket *avpkt) {
  avpkt->data         =payload.data(); // +4; that four is just for debugging..
  avpkt->size         =payload.size(); // -4;
  avpkt->stream_index =subsession_index;

  if (codec_id==AV_CODEC_ID_H264 and h264_pars.slice_type==H264SliceType::sps) { // we assume that frames always come in the following sequence: sps, pps, i, etc.
    avpkt->flags=AV_PKT_FLAG_KEY;
  }

  // std::cout << "Frame : useAVPacket : pts =" << pts << std::endl;

  if (mstimestamp>=0) {
    avpkt->pts=(int64_t)mstimestamp;
  }
  else {
    avpkt->pts=AV_NOPTS_VALUE;
  }

  // std::cout << "Frame : useAVPacket : final pts =" << pts << std::endl;

  avpkt->dts=AV_NOPTS_VALUE; // let muxer set it automagically 
}


void BasicFrame::copyFromAVPacket(AVPacket *pkt) {
  payload.resize(pkt->size);
  memcpy(payload.data(),pkt->data,pkt->size);
  // TODO: optimally, this would be done only once - in copy-on-write when writing to fifo, at the thread border
  subsession_index=pkt->stream_index;
  // frametype=FrameType::h264; // not here .. avpkt carries no information about the codec
  mstimestamp=(long int)pkt->pts;
}


void BasicFrame::filterFromAVPacket(AVPacket *pkt, AVCodecContext *codec_ctx, AVBitStreamFilterContext *filter) {
  int out_size;
  uint8_t *out;
  
  av_bitstream_filter_filter(
    filter,
    codec_ctx,
    NULL,
    &out,
    &out_size,
    pkt->data,
    pkt->size,
    pkt->flags & AV_PKT_FLAG_KEY
  );
  
  payload.resize(out_size);
  std::cout << "BasicFrame: filterFromAVPacket: " << out_size << " " << (long unsigned)(out) << std::endl; 
  memcpy(payload.data(),out,out_size);
  subsession_index=pkt->stream_index;
  mstimestamp=(long int)pkt->pts;
}


IdNumber BasicFrame::calcSize() {
    // device_id (std::size_t) subsession_index (int) mstimestamp (long int) media_type (AVMediaType) codec_id (AVCodecId) size (std::size_t) payload (char)
    // TODO: should use typedefs more
    return sizeof(IdNumber) + sizeof(subsession_index) + sizeof(mstimestamp) + sizeof(media_type) + sizeof(codec_id) + sizeof(std::size_t) + payload.size();
}

#define dump_bytes(var) os.write( (const char*)&var, sizeof(var));

bool BasicFrame::dump(IdNumber device_id, std::fstream &os) {
    std::size_t len;
    len=payload.size();
    
    dump_bytes(device_id);
    dump_bytes(subsession_index);
    dump_bytes(mstimestamp);
    dump_bytes(media_type);
    dump_bytes(codec_id);
    dump_bytes(len); // write the number of bytes
    os.write((const char*)payload.data(), payload.size()); // write the bytes themselves
    os.flush();
    return true;
}

#define read_bytes(var) is.read((char*)&var, sizeof(var));

IdNumber BasicFrame::read(std::fstream &is) {
    std::size_t len;
    IdNumber device_id;
    
    len = 0;
    device_id = 0;
    
    read_bytes(device_id);
    // is.read((char*)&device_id, sizeof(IdNumber));
    // std::cout << "BasicFrame : read : device_id =" << device_id << std::endl;
    
    if (device_id==0) { // no frame
        return device_id;
    }
    
    read_bytes(subsession_index);
    read_bytes(mstimestamp);
    read_bytes(media_type);
    read_bytes(codec_id);
    read_bytes(len);  // read the number of bytes
    payload.resize(len); 
    is.read((char*)payload.data(), len); // read the bytes themselves
    
    return device_id;
}


SetupFrame::SetupFrame() : Frame() {
  reset();
}
  
  
SetupFrame::~SetupFrame() {
}

//frame_essentials(FrameClass::setup, SetupFrame);

void SetupFrame::print(std::ostream &os) const {
  os << "<SetupFrame: timestamp="<<mstimestamp<<" subsession_index="<<subsession_index<<" slot="<<n_slot<<" / ";  //<<std::endl;
  os << "media_type=" << int(media_type) << " codec_id=" << int(codec_id);
  os << ">";
}


void SetupFrame::reset() {
  Frame::reset();
  media_type=AVMEDIA_TYPE_UNKNOWN;
  codec_id  =AV_CODEC_ID_NONE;
}



AVMediaFrame::AVMediaFrame() : media_type(AVMEDIA_TYPE_UNKNOWN), codec_id(AV_CODEC_ID_NONE) { // , mediatype(MediaType::none) {
  av_frame =av_frame_alloc();
}


AVMediaFrame::~AVMediaFrame() {
  av_frame_free(&av_frame);
  av_free(av_frame); // needs this as well?
}


//frame_essentials(FrameClass::avmedia, AVMediaFrame);


void AVMediaFrame::print(std::ostream& os) const {
  os << "<AVMediaFrame: timestamp="<<mstimestamp<<" subsession_index="<<subsession_index<<" slot="<<n_slot<<" / ";
  os << "media_type=" << media_type << std::endl;
  os << ">";
}


std::string AVMediaFrame::dumpPayload() {
  std::stringstream tmp;
  return tmp.str();
}


void AVMediaFrame::reset() {
  Frame::reset();
  media_type   =AVMEDIA_TYPE_UNKNOWN; 
  codec_id     =AV_CODEC_ID_NONE;
  // TODO: resert AVFrame ?
}



AVBitmapFrame::AVBitmapFrame() : AVMediaFrame(), av_pixel_format(AV_PIX_FMT_NONE), bmpars(BitmapPars()), y_payload(NULL), u_payload(NULL), v_payload(NULL) {
}


AVBitmapFrame::~AVBitmapFrame() {
}


//frame_essentials(FrameClass::avbitmap, AVBitmapFrame);
  

std::string AVBitmapFrame::dumpPayload() {
  std::stringstream tmp;  
  int i;
  for(i=0; i<std::min(bmpars.y_linesize,10); i++) {
    tmp << int(y_payload[i]) << " ";
  }
  return tmp.str();
}


void AVBitmapFrame::reset() {
  AVMediaFrame::reset();
  av_pixel_format =AV_PIX_FMT_NONE;
  bmpars          =BitmapPars();  
  y_payload       =NULL;       
  u_payload       =NULL;       
  v_payload       =NULL; 
}


void AVBitmapFrame::print(std::ostream& os) const {
  os << "<AVBitmapFrame: timestamp="<<mstimestamp<<" subsession_index="<<subsession_index<<" slot="<<n_slot<<" / ";
  os << "h=" << bmpars.height <<"; ";
  os << "w=" << bmpars.width  <<"; ";
  os << "l=("<< bmpars.y_linesize << "," << bmpars.u_linesize << "," << bmpars.v_linesize <<"); ";
  os << "f=" << av_pixel_format;
  os << ">";
}
 

void AVBitmapFrame::update() {
  const AVPixFmtDescriptor *desc =av_pix_fmt_desc_get(av_pixel_format);
  // const AVPixFmtDescriptor *desc =av_pix_fmt_desc_get(AV_PIX_FMT_YUV420P);

#ifdef DECODE_VERBOSE
  std::cout << "AVBitmapFrame: update: av_pixel_format   : " << int(av_pixel_format) << std::endl;
  std::cout << "AVBitmapFrame: update: width, desc factor: " << av_frame->width << ", " << desc->log2_chroma_w << std::endl;
#endif
  
  int w_fac =av_frame->width / AV_CEIL_RSHIFT(av_frame->width,  desc->log2_chroma_w);
  int h_fac =av_frame->height/ AV_CEIL_RSHIFT(av_frame->height, desc->log2_chroma_h);

#ifdef DECODE_VERBOSE
  std::cout << "AVBitmapFrame: update: w_fac, h_fac " << w_fac << ", " << h_fac << std::endl;
#endif
  
  bmpars=BitmapPars(
    0, 
    av_frame->width, 
    av_frame->height, 
    w_fac,
    h_fac
  );

  // image widths with paddings
  bmpars.y_linesize =av_frame->linesize[0];
  bmpars.u_linesize =av_frame->linesize[1];
  bmpars.v_linesize =av_frame->linesize[2];

#ifdef DECODE_VERBOSE
  std::cout << "AVBitmapFrame: update: bmpars= " << bmpars << std::endl;
#endif
  
  y_payload =av_frame->data[0];
  u_payload =av_frame->data[1];
  v_payload =av_frame->data[2];  
}



AVRGBFrame::AVRGBFrame() : AVBitmapFrame() {
}
  

AVRGBFrame::~AVRGBFrame() {
}


//frame_essentials(FrameClass::avrgb, AVRGBFrame);
 
  
std::string AVRGBFrame::dumpPayload() {
  std::stringstream tmp;  
  int i;
  for(i=0; i<std::min(bmpars.y_linesize,10); i++) {
    tmp << int(y_payload[i]) << " ";
  }
  return tmp.str();
}
  
  
void AVRGBFrame::print(std::ostream& os) const {
  os << "<AVRGBFrame: timestamp="<<mstimestamp<<" subsession_index="<<subsession_index<<" slot="<<n_slot<<" / ";
  os << "h=" << bmpars.height <<"; ";
  os << "w=" << bmpars.width  <<"; ";
  os << "l=("<< bmpars.y_linesize << "," << bmpars.u_linesize << "," << bmpars.v_linesize <<"); ";
  os << "f=" << av_pixel_format;
  os << ">";
}
  


/*
AVAudioFrame::AVAudioFrame() : AVMediaFrame(), av_sample_fmt(AV_SAMPLE_FMT_NONE) {
  mediatype=MediaType::audio;
}


AVAudioFrame::~AVAudioFrame() {
}


//frame_essentials(FrameClass::avaudio, AVAudioFrame);


std::string AVAudioFrame::dumpPayload() {
  return std::string("");
}


void AVAudioFrame::print(std::ostream& os) const {
  os << "<AVAudioFrame: timestamp="<<mstimestamp<<" subsession_index="<<subsession_index<<" slot="<<n_slot<<" / ";
  os << ">";
}


bool AVAudioFrame::isOK() {
  return true;
}


void AVAudioFrame::getParametersDecoder(const AVCodecContext *ctx) {
  av_media_type   = ctx->codec_type;
  av_sample_fmt   = ctx->sample_fmt;
  // TODO: from FFMpeg codecs to valkka codecs
  updateParameters();
}
*/



YUVFrame::YUVFrame(BitmapPars bmpars) : Frame(), bmpars(bmpars), y_index(0), u_index(0), v_index(0), y_payload(NULL), u_payload(NULL), v_payload(NULL) {
  reserve();
}


YUVFrame::~YUVFrame() {
  release();
}

//frame_essentials(FrameClass::yuv, YUVFrame);


void YUVFrame::reserve() {
  bool ok=true;
  
  getPBO(y_index,bmpars.y_size,y_payload);
  if (y_payload) {
    opengllogger.log(LogLevel::crazy) << "YUVFrame: reserve: Y: Got databuf pbo_id " <<y_index<< " with adr="<<(long unsigned)(y_payload)<<std::endl;
  } else {ok=false;}
  
  getPBO(u_index,bmpars.u_size,u_payload);
  if (u_payload) {
    opengllogger.log(LogLevel::crazy) << "YUVFrame: reserve: U: Got databuf pbo_id " <<u_index<< " with adr="<<(long unsigned)(u_payload)<<std::endl;
  } else {ok=false;}
  
  getPBO(v_index,bmpars.v_size,v_payload);
  if (v_payload) {
    opengllogger.log(LogLevel::crazy) << "YUVFrame: reserve: V: Got databuf pbo_id " <<v_index<< " with adr="<<(long unsigned)(v_payload)<<std::endl;
  } else {ok=false;}
  
  if (!ok) {
    // opengllogger.log(LogLevel::fatal) << "YUVFrame: reserve: WARNING: could not get GPU ram (out of memory?) "<<std::endl;
    perror("YUVFrame: reserve: WARNING: could not get GPU ram (out of memory?)");
  }
  
}


void YUVFrame::release() {
  opengllogger.log(LogLevel::crazy) << "YUVFrame: destructor"<<std::endl;
  releasePBO(&y_index, y_payload); y_payload=NULL;
  releasePBO(&u_index, u_payload); u_payload=NULL;
  releasePBO(&v_index, v_payload); v_payload=NULL;
  opengllogger.log(LogLevel::crazy) << "YUVFrame: destructor: bye"<<std::endl;
}
  
  
void YUVFrame::fromAVBitmapFrame(AVBitmapFrame *bmframe) {
  /* // av_image_copy_plane is used like this:
  av_image_copy_plane(
    uint8_t * dst,
    int dst_linesize,
    const uint8_t* src,
    int src_linesize,
    int bytewidth,
    int height 
  )
  // requires src_linesize[*] >= bytewidth ==> linesizes are widths with padding.  bytewidth is the true width
  */
  
  // YUVFrames are pre-reserved ..
  // So, here bmpars have the pre-reserved (maximum) dimensions.  They don't necessarily match with the dimensions of the frame we're uploading.
  // bmframe->bmpars has the actual dimensions to be uploaded
  // bmframe->linesizes have the optimal memcpy sizes (larger than or equal to dimensions in bmframe->bmpars), i.e. widths with paddings
  
  source_bmpars           =bmframe->bmpars;
  
  // TODO: check that source_bmpars <= bmpars
  if (bmpars.y_linesize<source_bmpars.y_linesize || bmpars.u_linesize<source_bmpars.u_linesize) {
    opengllogger.log(LogLevel::fatal) << "YUVFrame: fromAVBitmapFrame: WARNING: bitmap doesn't fit "<<std::endl;
    return;
  }
  
  const uint8_t* y_source =(uint8_t*)bmframe->y_payload;
  const uint8_t* u_source =(uint8_t*)bmframe->u_payload;
  const uint8_t* v_source =(uint8_t*)bmframe->v_payload;
  
#ifdef LOAD_VERBOSE
  std::cout << "YUVFrame: fromAVBitmapFrame target ptr : "<< (long unsigned)y_payload << " " << (long unsigned)u_payload << " " << (long unsigned)v_payload << " "<< std::endl;
  std::cout << "YUVFrame: fromAVBitmapFrame target y   : w, linesize, h : "<< bmpars.y_width << " " << bmpars.y_linesize << " " << bmpars.y_height << " "<< std::endl;
  std::cout << "YUVFrame: fromAVBitmapFrame target u   : w, linesize, h : "<< bmpars.u_width << " " << bmpars.u_linesize << " " << bmpars.u_height << " "<< std::endl;
  std::cout << "YUVFrame: fromAVBitmapFrame target v   : w, linesize, h : "<< bmpars.v_width << " " << bmpars.v_linesize << " " << bmpars.v_height << " "<< std::endl;
  std::cout << "YUVFrame: fromAVBitmapFrame source ptr : "<< (long unsigned)y_source << " " << (long unsigned)u_source << " " << (long unsigned)v_source << " "<< std::endl;
  std::cout << "YUVFrame: fromAVBitmapFrame source y   : w, linesize, h : "<< source_bmpars.y_width << " " << source_bmpars.y_linesize << " " << source_bmpars.y_height << " "<< std::endl;
  std::cout << "YUVFrame: fromAVBitmapFrame source u   : w, linesize, h : "<< source_bmpars.u_width << " " << source_bmpars.u_linesize << " " << source_bmpars.u_height << " "<< std::endl;
  std::cout << "YUVFrame: fromAVBitmapFrame source v   : w, linesize, h : "<< source_bmpars.v_width << " " << source_bmpars.v_linesize << " " << source_bmpars.v_height << " "<< std::endl;  
#endif
  
  av_image_copy_plane(
    y_payload,
    source_bmpars.y_width, // This is consistent what we're doing later on with the textures ..
    y_source,
    source_bmpars.y_linesize, // bmframe linesize >= bframe width
    source_bmpars.y_width,
    source_bmpars.y_height
  );
  
  av_image_copy_plane(
    u_payload,
    source_bmpars.u_width,
    u_source,
    source_bmpars.u_linesize, // bmframe linesize >= bframe width
    source_bmpars.u_width,
    source_bmpars.u_height
  );
  
  av_image_copy_plane(
    v_payload,
    source_bmpars.v_width,
    v_source,
    source_bmpars.v_linesize, // bmframe linesize >= bframe width
    source_bmpars.v_width,
    source_bmpars.v_height
  );
  
  // std::cout << this->dumpPayload() << std::endl;
  
  copyMetaFrom(bmframe);
  
  /* // old crust
  memcpy(y_payload, bmframe->y_payload, std::min(bmframe->y_size,y_size));
  memcpy(u_payload, bmframe->u_payload, std::min(bmframe->u_size,u_size)); 
  memcpy(v_payload, bmframe->v_payload, std::min(bmframe->v_size,v_size));
  */
#ifdef LOAD_VERBOSE
  std::cout << "YUVFrame: fromAVBitmapFrame done" << std::endl;
#endif
}


std::string YUVFrame::dumpPayload() {
  std::stringstream tmp;
  int i;

  tmp << "[";
  for(i=0; i<std::min(10,bmpars.y_size); i++) {
    tmp << (unsigned int)y_payload[i] << " ";
  }
  tmp << "] ";
  
  tmp << "[";
  for(i=0; i<std::min(10,bmpars.u_size); i++) {
    tmp << (unsigned int)u_payload[i] << " ";
  }
  tmp << "] ";
  
  tmp << "[";
  for(i=0; i<std::min(10,bmpars.v_size); i++) {
    tmp << (unsigned int)v_payload[i] << " ";
  }
  tmp << "] ";
  
  return tmp.str();  
}


void YUVFrame::print(std::ostream& os) const {
  os << "<YUVFrame: timestamp="<<mstimestamp<<" subsession_index="<<subsession_index<<" slot="<<n_slot<<" / ";
  os << "y_ptr=" << (unsigned long)y_payload << " " << "u_ptr=" << (unsigned long)u_payload << " " << "v_ptr=" << (unsigned long)v_payload;
  os << ">";
}

  
void YUVFrame::reset() {
  Frame::reset();
  source_bmpars =BitmapPars();
}



SignalFrame::SignalFrame() : Frame(), opengl_signal_ctx(), av_signal_ctx(), custom_signal_ctx(NULL) {
  mstimestamp =getCurrentMsTimestamp();
}
  
SignalFrame::~SignalFrame() {
}


//frame_essentials(FrameClass::signal, SignalFrame);


