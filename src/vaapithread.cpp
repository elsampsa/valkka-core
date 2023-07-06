/*
 * vaapithread.cpp :
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
 *  @file    vaapithread.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
 *  
 *  @brief 
 */ 

#include "vaapithread.h"


VAAPIThread::VAAPIThread(const char* name, FrameFilter& outfilter, FrameFifoContext fifo_ctx) 
    : DecoderThread(name, outfilter, fifo_ctx)
    {
    }

VAAPIThread::~VAAPIThread() {
}

Decoder* VAAPIThread::chooseAudioDecoder(AVCodecID codec_id) {
    DecoderThread::chooseAudioDecoder(codec_id);
}

Decoder* VAAPIThread::chooseVideoDecoder(AVCodecID codec_id) {
    //TODO: try to get NVDecoder, if it's not possible, default
    //to AVDecoder
    switch (codec_id) { // switch: video codecs
        case AV_CODEC_ID_H264:
            return new HwVideoDecoder(AV_CODEC_ID_H264);
            break;
        default:
            return NULL;
            break;        
    }
}

Decoder* VAAPIThread::fallbackAudioDecoder(AVCodecID codec_id) {
    return DecoderThread::chooseAudioDecoder(codec_id);
}


Decoder* VAAPIThread::fallbackVideoDecoder(AVCodecID codec_id) {
    return DecoderThread::chooseVideoDecoder(codec_id);
}
