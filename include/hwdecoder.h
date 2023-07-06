#ifndef AVHwDecoder_HEADER_GUARD
#define AVHwDecoder_HEADER_GUARD
/*
 * AVHwDecoder.h :
 * 
 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
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
 *  @file    AVHwDecoder.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
 *  
 *  @brief
 */ 

#include "decoder.h"
extern "C" {
#include <libavutil/hwcontext.h>
}

static enum AVPixelFormat get_vaapi_format(AVCodecContext *ctx,
                                           const enum AVPixelFormat *pix_fmts)
{
    const enum AVPixelFormat *p;
 
    for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == AV_PIX_FMT_VAAPI)
            return *p;
    }
 
    fprintf(stderr, "Unable to decode this file using VA-API.\n");
    return AV_PIX_FMT_NONE;
}

/** Video decoder using FFmpeg/libav with VAAPI
 * 
 * Subclassed from the basic ffmpeg/libav decoder class VideoDecoder
 * 
 * See also \ref pipeline
 * 
 * @ingroup decoding_tag
 * 
 * TODO: 
 * from here we still need to subclass into HWVideoDecoder
 * i.e. to define Frame *output() 
 * bool pull()
 * 
 * like class VideoDecoder : public AVDecoder
 * 
 */
class AVHwDecoder : public Decoder
{

public:
    AVHwDecoder(AVCodecID av_codec_id, int n_threads = 1); ///< Default constructor
    virtual ~AVHwDecoder();                                ///< Default destructor

protected:
    int n_threads;
    bool active;

public:
    AVCodecID av_codec_id;            ///< FFmpeg AVCodecId, identifying the codec
    AVPacket *av_packet;              ///< FFmpeg internal data structure; encoded frame (say, H264)
    AVCodecContext *av_codec_context; ///< FFmpeg internal data structure
    AVCodec *av_codec;                ///< FFmpeg internal data structure
    AVBufferRef* hw_device_ctx;       ///< FFmpeg/libav hardware context

public:
    // needs virtual void output, virtual void pull
    virtual void flush();
    virtual bool isOk();

};

class HwVideoDecoder : public AVHwDecoder
{

public:
    HwVideoDecoder(AVCodecID av_codec_id, int n_threads = 1); ///< Default constructor
    virtual ~HwVideoDecoder();                                ///< Default destructor

protected:
    AVBitmapFrame out_frame;
    int width;
    int height;
    AVFrame *aux_av_frame;
    AVPixelFormat current_pixel_format;
    SwsContext *sws_ctx;
    float secs_per_frame;

public:
    virtual Frame *output(); ///< Return a reference to the internal storage of the decoder where the decoded frame is
    virtual bool pull();

};



#endif
