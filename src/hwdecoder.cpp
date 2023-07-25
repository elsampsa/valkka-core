/*
 * hwdecoder.cpp :
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
 *  @file    hwdecoder.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
 *
 *  @brief
 */

#include "hwdecoder.h"

// https://ffmpeg.org/doxygen/trunk/vaapi_transcode_8c-example.html#a66
// we need this..?
// https://gist.github.com/kajott/d1b29c613be30893c855621edd1f212e
// this is it:
// https://ffmpeg.org/doxygen/3.4/hw_decode_8c-example.html

AVHwDecoder::AVHwDecoder(AVCodecID av_codec_id, AVHWDeviceType hwtype, int n_threads) : av_codec_id(av_codec_id), n_threads(n_threads), active(true)
{
    // http://ffmpeg.org/doxygen/3.4/hwcontext_8c.html
    // TODO: AVBufferRef *hw_device_ctx
    int ret = av_hwdevice_ctx_create(&hw_device_ctx, hwtype, NULL, NULL, 0);
    if (ret < 0)
    {
        avthreadlogger.log(LogLevel::fatal) << "AVHwDecoder: failed to create a hw device. Error: " << av_err2str(ret) << std::endl;
        active = false;
    }
    int retcode;
    has_frame = false;
    // av_packet       =av_packet_alloc(); // oops.. not part of the public API (defined in avcodec.c instead of avcodec.h)
    av_packet = new AVPacket();
    av_init_packet(av_packet);
    hw_frame = av_frame_alloc();      // TODO: free in dtor
    // cpu_aux_frame = av_frame_alloc(); // TODO: free in dtor
    av_codec = avcodec_find_decoder(av_codec_id);
    if (!av_codec)
    {
        std::perror("AVHwDecoder: FATAL: could not find av_codec");
        exit(2);
    }
    av_codec_context = avcodec_alloc_context3(av_codec);

    // if (false) { //debug: skip hw context etc. creation
    if (active) {
        avthreadlogger.log(LogLevel::normal) << "AVHwDecoder: attaching hardware context" << std::endl;
        av_codec_context->hw_device_ctx = av_buffer_ref(hw_device_ctx);
        av_codec_context->get_format = get_vaapi_format;
        hw_pix_format = find_fmt_by_hw_type(hwtype);
        if (!av_codec_context->hw_device_ctx)
        {
            avthreadlogger.log(LogLevel::fatal) << "AVHwDecoder: a hardware device reference create failed " << std::endl;
            active = false;
        }
        else
        {
            /*
            // final step: link AVFrame hardware frame context
            // to hardware context
            hw_frame->hw_frames_ctx =
                av_hwframe_ctx_alloc(av_codec_context->hw_device_ctx);
            if (!hw_frame->hw_frames_ctx) {
                avthreadlogger.log(LogLevel::fatal) << "AVHwDecoder: linking hardware frames to hardware device failed " << std::endl;
                active=false;
            }
            */
        }
    }
    else
    {
        // av_codec_context->get_format = get_vaapi_format;
    }

    ///*
    av_codec_context->thread_count = this->n_threads;
    if (this->n_threads > 1)
    {
        decoderlogger.log(LogLevel::debug) << "AVHwDecoder: using multithreading with " << this->n_threads << std::endl;
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

AVHwDecoder::~AVHwDecoder()
{
    if (active)
    {
        av_buffer_unref(&hw_device_ctx);
    }
    // https://stackoverflow.com/questions/14914462/ffmpeg-memory-leak
    av_free_packet(av_packet);
    // av_frame_free(&av_frame);
    // av_free(av_frame); // needs this as well?
    avcodec_close(av_codec_context);
    avcodec_free_context(&av_codec_context);
    delete av_packet;
}

void AVHwDecoder::flush()
{
    avcodec_flush_buffers(av_codec_context);
}

bool AVHwDecoder::isOk()
{
    return active;
}

// TODO:

// this one!
// https://ffmpeg.org/doxygen/3.4/hw_decode_8c-example.html

// https://ffmpeg.org/doxygen/3.4/hwcontext_8h.html#abf1b1664b8239d953ae2cac8b643815a
/*

int av_hwframe_transfer_data 	( 	AVFrame *  	dst,
        const AVFrame *  	src,
        int  	flags
    )

Copy data to or from a hw surface.
At least one of dst/src must have an AVHWFramesContext attached.

AVBufferRef* AVFrame::hw_frames_ctx
For hwaccel-format frames, this should be a reference to the AVHWFramesContext describing the frame

https://ffmpeg.org/doxygen/3.4/structAVHWFramesContext.html#details

This struct describes a set or pool of "hardware" frames (i.e. those with data not located in normal system memory).
All the frames in the pool are assumed to be allocated in the same way and interchangeable.

This struct is reference-counted with the AVBuffer mechanism and tied to a given AVHWDeviceContext instance.
The av_hwframe_ctx_alloc() constructor yields a reference, whose data field points to the actual AVHWFramesContext struct.

AVBufferRef* av_hwframe_ctx_alloc 	( 	AVBufferRef *  	device_ctx	)


av_hwdevice_ctx_create(&hw_device_ctx, hwtype, NULL, NULL, 0);
hw_frame_ctx = av_hwframe_ctx_alloc(hw_device_ctx);
AVFrame* hw_frame
hw_frame->hw_frames_ctx = hw_frame_ctx

or

hw_frame->hw_frames_ctx = av_hwframe_ctx_alloc(hw_device_ctx)

.. so now AVFrame* hw_frame has hwframe context attached
*/

HwVideoDecoder::HwVideoDecoder(AVCodecID av_codec_id, AVHWDeviceType hwtype, int n_threads) : 
    AVHwDecoder(av_codec_id, hwtype, n_threads), width(0), height(0),
    current_pixel_format(AV_PIX_FMT_YUV420P), sws_ctx(NULL) {
        aux_av_frame = av_frame_alloc();
    }
                                
HwVideoDecoder::~HwVideoDecoder()
{
    if (sws_ctx != NULL)
    {
        sws_freeContext(sws_ctx);
        sws_ctx = NULL;
    }
    av_frame_free(&aux_av_frame);
    av_free(aux_av_frame); // need this?
};

Frame *HwVideoDecoder::output()
{
    return &out_frame;
};

bool HwVideoDecoder::pull()
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

    if (av_codec_context->framerate.num > 0)
    {
        av_packet->pts = (av_codec_context->framerate.num * in_frame.mstimestamp) / 1000;
        // std::cout << "av_packet->pts :" << av_packet->pts << std::endl;
    }

#ifdef DECODE_VERBOSE
    std::cout << "HwVideoDecoder: pull: size    =" << av_packet->size << std::endl;
    std::cout << "HwVideoDecoder: pull: payload =[" << in_frame.dumpPayload() << "]" << std::endl;
#endif

#ifdef DECODE_TIMING
    long int mstime = getCurrentMsTimestamp();
#endif

    std::cout << "send_packet" << std::endl;
    retcode = avcodec_send_packet(av_codec_context, av_packet);
    if (retcode < 0)
    {
        // fprintf(stderr, "Error during decoding (0)\n");
        std::cout << "Error during decoding (0): " << retcode << " -- > " << av_err2str(retcode) << std::endl;
        // this simply means that the packet is unexpected, i.e. a live stream that starts without
        // SPS and PPS packets, but that will eventually be corrected
        // active=false;
        return false;
    }
    retcode = avcodec_receive_frame(av_codec_context, hw_frame);
    if (retcode < 0)
    {
        fprintf(stderr, "Error during decoding (1)\n");
        active = false;
        return false;
    }

    // retcode = avcodec_decode_video2(av_codec_context,
    //     hw_frame, &got_frame, av_packet);

    if (hw_frame->format == hw_pix_format)
    {
        /* retrieve data from GPU to CPU */
        std::cout << "correct hw pix fmt found" << std::endl;
    }
    else
    {
        std::cout << "unexpected hw pix fmt" << std::endl;
        active = false;
        return false;
    }
    
    AVFrame *av_ref_frame;                      // source data: should go through pix tf or not
    AVFrame *out_av_frame = out_frame.av_frame; // shorthand ref: final outgoing frame

    std::cout << "current pix fmt " << current_pixel_format << std::endl;

    std::cout << "HW COPY" << std::endl;

    if (current_pixel_format == AV_PIX_FMT_YUV420P)
    {
        // so, imagining that a hw device would give us directly yuv420p.. will never happend!
        // but let's follow the same "logic" as before. :)
        retcode = av_hwframe_transfer_data(
            out_av_frame,            // AVFrame* dst
            hw_frame,                // AVFrame * src
            0);                      // that typically spits out AV_PIX_FMT_NV12
        av_ref_frame = out_av_frame; // that's it!
    }
    else
    {   // NOTE: should not and will never end here upon arrival of the first frame
        std::cout << "transferring to aux_av_frame" << std::endl;
        retcode = av_hwframe_transfer_data(
            aux_av_frame, // AVFrame* dst
            hw_frame,     // AVFrame * src
            0);           // that typically spits out AV_PIX_FMT_NV12
        av_ref_frame = aux_av_frame;
        // we need to go through pix transformation
    }

    std::cout << "format       : " << av_ref_frame->format << std::endl;
    std::cout << "dims         : " << av_ref_frame->width << "x" << av_ref_frame->height << std::endl;
    std::cout << "linesizes    : " << av_ref_frame->linesize[0] << " " << av_ref_frame->linesize[1] << " " << av_ref_frame->linesize[2] << " " 
        << std::endl;
    std::cout << "hw_linesizes : " << hw_frame->linesize[0] << " " << hw_frame->linesize[1] << " " << hw_frame->linesize[2] << " " 
        << std::endl;


    // std::cout << "vaapi:" << hw_pix_format << ", nv12:" << AV_PIX_FMT_NV12 << ", 420p:" << AV_PIX_FMT_YUV420P << std::endl;
    // vaapi:53, nv12:25, 420p:0

    if (retcode < 0)
    {
        fprintf(stderr, "Error transferring data to CPU\n");
        active = false;
        return false;
    }

    std::cout << "HW COPY END" << std::endl;

    if (av_codec_context->framerate.num > 0)
    {
        // std::cout << "av_ref_frame->pts :" << av_ref_frame->pts << std::endl;
        pts = 1000 * av_ref_frame->pts / av_codec_context->framerate.num;
        // no problem here.. thousand always divides nicely with
        // a sane fps number (say, with 20)
    }

    if (av_ref_frame->width < 1 or av_ref_frame->height < 1)
    { // crap
#ifdef DECODE_VERBOSE
        std::cout << "HwVideoDecoder: pull: corrupt frame " << std::endl;
#endif
        std::cout << "HwVideoDecoder: pull: corrupt frame " << std::endl;
        return false;
    }

    // check the current pixel format
    AVPixelFormat new_pixel_format = static_cast<AVPixelFormat>(av_ref_frame->format);

    if (new_pixel_format == AV_PIX_FMT_YUVJ420P)
    {
        new_pixel_format = AV_PIX_FMT_YUV420P;
    }

    if (new_pixel_format != current_pixel_format)
    { // PIX FMT CHANGE
        // pixel format has changed:
        // this is due to the (assumed) initial condition of AV_PIX_FMT_YUV420P
        // or because of the device using this decoder has been changed
        // or because the device has changed the pix format itself
        if (sws_ctx != NULL)
        {
            sws_freeContext(sws_ctx);
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
            // https://stackoverflow.com/questions/50023750/how-to-convert-from-av-pix-fmt-vaapi-to-av-pix-fmt-yuv420-using-ffmpeg-sws-scale
            /*
            SwsContext* conversion_context_ = sws_getContext(videoDecoder_->width(),
                videoDecoder_->height(), AV_PIX_FMT_NV12 ,scaler_->getWidth(), scaler_->getHeight(), AV_PIX_FMT_YUV420P,(int)SWS_BICUBIC,
            nullptr, nullptr, nullptr);
            */
            sws_ctx = sws_getContext(
                av_ref_frame->width, av_ref_frame->height,
                new_pixel_format,
                av_ref_frame->width, av_ref_frame->height,
                AV_PIX_FMT_YUV420P,
                SWS_POINT, // should be the best for byte-shuffling only
                // SWS_FAST_BILINEAR,
                // SWS_BICUBIC,
                NULL, NULL, NULL);
            decoderlogger.log(LogLevel::normal)
                << "HwVideoDecoder: WARNING: pix re-fmt from " << current_pixel_format <<" to "<< new_pixel_format << std::endl;
            // the frame was decoded directly to out_frame.av_frame, assuming it was YUV420P
            // so an extra step is needed (only once):
            /*
            if (aux_av_frame != av_ref_frame) { // pointer comparison
                std::cout << "swap copy" << std::endl;
                av_frame_copy(aux_av_frame, av_ref_frame); // dst, src
            }
            */
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
                    32                       //align
                    );
            */
            std::cout << "sws_getContext: " << av_ref_frame->width << "x" << av_ref_frame->height << std::endl;
            std::cout << "sws_getContext: aux_frame: " << aux_av_frame->width << "x" << aux_av_frame->height << std::endl;
            out_frame.reserve(av_ref_frame->width, av_ref_frame->height); // out_av_frame -> out_frame.av_frame
            return false;
        }
    } // PIX FMT CHANGE

    if (sws_ctx != NULL) // sws_scale is needed!
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
        std::cout << "sws_scale!" << std::endl;

        int xs = aux_av_frame->width;
        /* // no aligment! see constant.h
        int mod = xs%32;
        if (mod>0) { // correct to nearest (higher) multiple of 32
            xs += 32-mod;
        }
        */
        // target linesizes
        out_av_frame->linesize[0]=xs;
        out_av_frame->linesize[1]=xs/2;
        out_av_frame->linesize[2]=xs/2;

        std::cout << "sws_scale linesizes : " << out_av_frame->linesize[0] << " " << out_av_frame->linesize[1] << " " << out_av_frame->linesize[2] << " " 
        << std::endl;

        height = sws_scale(
            sws_ctx,
            (const uint8_t *const *)aux_av_frame->data, // srcSlice[]
            aux_av_frame->linesize,                     // srcStride
            0,                                          // srcSliceY
            aux_av_frame->height,                       // srcSliceH
            out_av_frame->data,                         // dst[] // written
            out_av_frame->linesize);                    // dstStride[] // must provide!

        // for h=720; w=1280; strides should look like this: 1280,640,640
        // but it looks like: 1280,1280,0
    }

    if (height != out_av_frame->height || width != out_av_frame->width)
    {                                                   // update aux objects only if needed
        out_frame.av_pixel_format = AV_PIX_FMT_YUV420P; // converted: always YUV420P
        out_frame.updateAux();                          // uses av_frame and out_frame.av_pixel_format
        height = out_av_frame->height;
        width = out_av_frame->width;
        std::cout << "AUX UPDATE" << std::endl;
    }
    out_frame.update();
    std::cout << "UPDATE" << std::endl;
    out_frame.copyMetaFrom(&in_frame); // after this, the AVBitmapFrame instance is ready to go..
    // std::cout << "mstimestamps: " << out_frame.mstimestamp << " " << pts << std::endl;
    if (pts > 0)
    {
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
        std::cout << "HwVideoDecoder: pull: decoding took " << mstime << "milliseconds" << std::endl;
    }
#endif

    /*
  // debugging: let's simulate slow decoding
  int delay=dis(gen)+40;
  std::cout << "HwVideoDecoder: pull: delay=" << delay << std::endl;
  std::this_thread::sleep_for(std::chrono::milliseconds(delay));
  */

#ifdef DECODE_VERBOSE
    std::cout << "HwVideoDecoder: pull: retcode, got_frame, AVBitmapFrame : " << retcode << " " << got_frame << " " << out_frame << std::endl;
#endif

#ifdef DECODE_VERBOSE
    std::cout << "HwVideoDecoder: decoded: " << retcode << " bytes => " << out_frame << std::endl;
    std::cout << "HwVideoDecoder: payload: " << out_frame.dumpPayload() << std::endl;
#endif

    has_frame = true;
    return true;
}
