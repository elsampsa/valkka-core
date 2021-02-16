/*
 * decoders.cpp : FFmpeg decoders
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
 *  @file    decoders.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.1.0 
 *  @brief   FFmpeg decoders
 */

#include "decoder.h"
#include "logging.h"

// https://stackoverflow.com/questions/14914462/ffmpeg-memory-leak
// #define AV_REALLOC

// #define DECODE_VERBOSE

Decoder::Decoder() : has_frame(false) {
} ;

Decoder::~Decoder(){};

long int Decoder::getMsTimestamp()
{
    return in_frame.mstimestamp;
}

void Decoder::input(Frame *f)
{
    if (f->getFrameClass() != FrameClass::basic)
    {
        decoderlogger.log(LogLevel::normal) << "Decoder: input: can only accept BasicFrame" << std::endl;
    }
    else
    {
        in_frame.copyFrom(f);
    }
}

bool Decoder::hasFrame()
{
    return has_frame;
}

void Decoder::releaseOutput()
{
    // by default, does nada
};

bool Decoder::isOk() 
{
    return true;
}



Frame *DummyDecoder::output()
{
    return &out_frame;
}

void DummyDecoder::flush(){};

bool DummyDecoder::pull()
{
    out_frame = in_frame;
    has_frame = true;
    return true;
}

AVDecoder::AVDecoder(AVCodecID av_codec_id, int n_threads) : Decoder(), av_codec_id(av_codec_id), n_threads(n_threads)
{
    int retcode;

    has_frame = false;

    // av_packet       =av_packet_alloc(); // oops.. not part of the public API (defined in avcodec.c instead of avcodec.h)
    av_packet = new AVPacket();
    av_init_packet(av_packet);
    // av_frame        =av_frame_alloc();

    // std::cout << "AVDecoder : AV_CODEC_ID=" << av_codec_id << " AV_CODEC_ID_H264=" << AV_CODEC_ID_H264 << " AV_CODEC_ID_INDEO3=" << AV_CODEC_ID_INDEO3 <<std::endl;

    av_codec = avcodec_find_decoder(av_codec_id);
    if (!av_codec)
    {
        std::perror("AVDecoder: FATAL: could not find av_codec");
        exit(2);
    }
    av_codec_context = avcodec_alloc_context3(av_codec);

    ///*
    av_codec_context->thread_count = this->n_threads;
    if (this->n_threads > 1)
    {
        std::cout << "AVDecoder: using multithreading with : " << this->n_threads << std::endl;
        decoderlogger.log(LogLevel::debug) << "AVDecoder: using multithreading with " << this->n_threads << std::endl;
        // av_codec_context->thread_type = FF_THREAD_SLICE; // decode different parts of the frame in parallel // FF_THREAD_FRAME = decode several frames in parallel
        // woops .. I get "data partition not implemented" for this version of ffmpeg (3.4)
        av_codec_context->thread_type = FF_THREAD_FRAME; // now frames come in bursts .. think about this
    }
    else
    {
        av_codec_context->thread_type = FF_THREAD_FRAME;
    }
    //*/

    // av_codec_context->refcounted_frames = 1; // monitored the times av_buffer_alloc was called .. makes no different really
    retcode = avcodec_open2(av_codec_context, av_codec, NULL);

    // is this going to change at some moment .. ? https://blogs.gentoo.org/lu_zero/2016/03/29/new-avcodec-api/
    // aha..! it's here: https://ffmpeg.org/doxygen/3.2/group__lavc__decoding.html#ga58bc4bf1e0ac59e27362597e467efff3
    if (retcode != 0)
    {
        std::perror("AVDecoder: FATAL could not open codec");
        exit(2);
    }
    else
    {
        decoderlogger.log(LogLevel::debug) << "AVDecoder: registered decoder with av_codec_id " << av_codec_id << std::endl;
    }

}

AVDecoder::~AVDecoder()
{
    // https://stackoverflow.com/questions/14914462/ffmpeg-memory-leak
    av_free_packet(av_packet);

    // av_frame_free(&av_frame);
    // av_free(av_frame); // needs this as well?

    avcodec_close(av_codec_context);
    avcodec_free_context(&av_codec_context);

    delete av_packet;
}

void AVDecoder::flush()
{
    avcodec_flush_buffers(av_codec_context);
}

VideoDecoder::VideoDecoder(AVCodecID av_codec_id, int n_threads) : 
AVDecoder(av_codec_id, n_threads), width(0), height(0), 
    current_pixel_format(AV_PIX_FMT_YUV420P), sws_ctx(NULL) {
    /*
    // decoder slow down simulation
    gen =std::mt19937(rd());
    dis =std::uniform_int_distribution<>(0,5);
    */

   // timebase
   // num = 25
   // den = 1

/*
   timebase = av_make_q(1000, 1)

   AVRational val;
   val.num = 1000;
   val.den = 1;
   av_codec_set_pkt_timebase(av_codec_context, val);
   */
};


VideoDecoder::~VideoDecoder(){
    if (sws_ctx != NULL)
    {
        sws_freeContext(sws_ctx);
        sws_ctx = NULL;
        av_frame_free(&aux_av_frame);
        av_free(aux_av_frame); // need this?
    }
};


Frame *VideoDecoder::output()
{
    return &out_frame;
};


bool VideoDecoder::pull()
{
    int retcode;
    int got_frame;
    long int pts = 0;

    has_frame = false;

    /* // some debugging .. (filter just sps, pps and keyframes)
    if (in_frame.h264_pars.slice_type != H264SliceType::pps and in_frame.h264_pars.slice_type != H264SliceType::sps and in_frame.h264_pars.slice_type != H264SliceType::i) {
        return false;
    }
  */

    av_packet->data = in_frame.payload.data(); // pointer to payload
    av_packet->size = in_frame.payload.size();

    // std::cout << "num:" << av_codec_context->framerate.num << std::endl;

    if (av_codec_context->framerate.num > 0) {
        av_packet->pts = (av_codec_context->framerate.num * in_frame.mstimestamp) / 1000;
        // std::cout << "av_packet->pts :" << av_packet->pts << std::endl;
    }

    
#ifdef DECODE_VERBOSE
    std::cout << "VideoDecoder: pull: size    =" << av_packet->size << std::endl;
    std::cout << "VideoDecoder: pull: payload =[" << in_frame.dumpPayload() << "]" << std::endl;
#endif

#ifdef DECODE_TIMING
    long int mstime = getCurrentMsTimestamp();
#endif

    /*
#ifdef AV_REALLOC
  av_frame_free(&av_frame);
  av_frame =av_frame_alloc();
#endif
*/

    AVFrame *av_ref_frame;
    AVFrame *out_av_frame = out_frame.av_frame; // shorthand ref

    if (current_pixel_format == AV_PIX_FMT_YUV420P) {
        retcode = avcodec_decode_video2(av_codec_context, out_frame.av_frame, &got_frame, av_packet);
        av_ref_frame = out_frame.av_frame;
    }
    else {
        // std::cout << "PIX_FMT" << std::endl;
        retcode = avcodec_decode_video2(av_codec_context, aux_av_frame, &got_frame, av_packet);
        av_ref_frame = aux_av_frame;
    }

    if (av_codec_context->framerate.num > 0) {
        // std::cout << "av_ref_frame->pts :" << av_ref_frame->pts << std::endl;
        pts = 1000 * av_ref_frame->pts / av_codec_context->framerate.num;
        // no problem here.. thousand always divides nicely with
        // a sane fps number (say, with 20)
    }

    if (retcode < 0)
    {
        decoderlogger.log(LogLevel::crazy) << "VideoDecoder: decoder error " << retcode << std::endl;
        return false;
    }

    if (av_ref_frame->width < 1 or av_ref_frame->height < 1)
    { // crap
#ifdef DECODE_VERBOSE
        std::cout << "VideoDecoder: pull: corrupt frame " << std::endl;
#endif
        return false;
    }
    /*
    retcode = avcodec_send_packet(av_codec_context, av_packet); // new API
    avcodec_receive_frame(av_codec_context, out_frame.av_frame); // new API
    */

    AVPixelFormat &new_pixel_format = av_codec_context->pix_fmt;

    if (new_pixel_format == AV_PIX_FMT_YUVJ420P) {
        new_pixel_format = AV_PIX_FMT_YUV420P;
    }

    if (new_pixel_format != current_pixel_format) 
    {
        if (sws_ctx != NULL)
        {
            sws_freeContext(sws_ctx);
            av_frame_free(&aux_av_frame);
            av_free(aux_av_frame);
            sws_ctx = NULL; // implies that no yuv conversion is needed
        }

        if (new_pixel_format == AV_PIX_FMT_YUV420P)
        { 
            // ok
            // special case: pix fmt suddenly changed from whatever to the default pixel format
            // then, let's just miss a frame (a very rare case)
            current_pixel_format = new_pixel_format;
        }
        else 
        {
            sws_ctx =sws_getContext(
                out_av_frame->width, out_av_frame->height, 
                new_pixel_format, 
                out_av_frame->width, out_av_frame->height, 
                AV_PIX_FMT_YUV420P, 
                // SWS_POINT, 
                SWS_FAST_BILINEAR,
                NULL, NULL, NULL);
            decoderlogger.log(LogLevel::normal) 
                << "VideoDecoder: WARNING: scaling your strange YUV format to YUV420P. " 
                << "This will be inefficient!"
                << std::endl;

            aux_av_frame =av_frame_alloc();
            // the frame was decoded directly to out_frame.av_frame, assuming it was YUV420P
            // so an extra step is needed (only once):
            av_frame_copy(aux_av_frame, out_av_frame); // dst, src
            // now the algorithm can process as normal
            current_pixel_format = new_pixel_format;

            // normally out_av_frame allocation is handled
            // by avcodec_decode_2
            // av_frame_free(&out_av_frame);
            // av_free(out_av_frame);
            /*
            av_image_alloc( (*rgbPictInfo)->data,   //data to be filled
                    (*rgbPictInfo)->linesize,//line sizes to be filled
                    width, height,
                    AV_PIX_FMT_YUV420P,           //pixel format
                    32                       //aling
                    );
            */
           out_frame.reserve(out_av_frame->width, out_av_frame->height);
        }
    }

    if (sws_ctx != NULL) // implies that there has been sws_scaling
    {
        /*
        int sws_scale 	( 	struct SwsContext *  	c,
		const uint8_t *const  	srcSlice[],
		const int  	srcStride[],
		int  	srcSliceY,
		int  	srcSliceH,
		uint8_t *const  	dst[],
		const int  	dstStride[] 
	    ) 	
        */
       // std::cout << "sws_scale!" << std::endl;
        height = sws_scale(sws_ctx, 
            (const uint8_t * const*)aux_av_frame->data,  // srcSlice[]
            aux_av_frame->linesize, // srcStride
            0,  // srcSliceY
            aux_av_frame->height,  // srcSliceH
            out_av_frame->data, // dst[] // written
            out_av_frame->linesize); // dstStride[] // written
    }
    

    if (height != out_av_frame->height || width != out_av_frame->width)
    { // update aux objects only if needed
        out_frame.av_pixel_format = AV_PIX_FMT_YUV420P; // converted: always YUV420P
        out_frame.updateAux(); // uses av_frame and out_frame.av_pixel_format
        height = out_av_frame->height;
        width = out_av_frame->width;
        // std::cout << "AUX UPDATE" << std::endl;
    }
    out_frame.update();
    // std::cout << "UPDATE" << std::endl;
    out_frame.copyMetaFrom(&in_frame); // after this, the AVBitmapFrame instance is ready to go..
    // std::cout << "mstimestamps: " << out_frame.mstimestamp << " " << pts << std::endl;
    if (pts > 0) {
        /*
        std::cout << "mstimestamps: " 
        << out_frame.mstimestamp << " "
        << pts << " "
        << pts - getCurrentMsTimestamp() << " " 
        << std::endl;
        */
        out_frame.mstimestamp = pts;
    }
    // out_frame.mstimestamp = out_av_frame->pts; // WARNING: TODO: must fix this.. otherwise streams using B-frames do not work at all!
    // timestamp is now the presentation timestamp given by the decoder 
    // .. uh-oh need to convert from time_base units of AVStream

    /*
#ifdef AV_REALLOC
  av_free_packet(av_packet);
  av_init_packet(av_packet);
#endif
*/

#ifdef DECODE_TIMING
    mstime = getCurrentMsTimestamp() - mstime;
    if (mstime > 40)
    {
        std::cout << "VideoDecoder: pull: decoding took " << mstime << "milliseconds" << std::endl;
    }
#endif

    /*
  // debugging: let's simulate slow decoding
  int delay=dis(gen)+40;
  std::cout << "VideoDecoder: pull: delay=" << delay << std::endl;
  std::this_thread::sleep_for(std::chrono::milliseconds(delay));
  */

#ifdef DECODE_VERBOSE
    std::cout << "VideoDecoder: pull: retcode, got_frame, AVBitmapFrame : " << retcode << " " << got_frame << " " << out_frame << std::endl;
#endif

#ifdef DECODE_VERBOSE
    std::cout << "VideoDecoder: decoded: " << retcode << " bytes => " << out_frame << std::endl;
    std::cout << "VideoDecoder: payload: " << out_frame.dumpPayload() << std::endl;
#endif

    has_frame = true;
    return true;
}
