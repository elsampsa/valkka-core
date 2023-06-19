/*
 * avthread.cpp : FFmpeg decoding thread
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
 *  @file    avthread.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.3.6 
 *  @brief   FFmpeg decoding thread
 */ 

#include "avthread.h"
#include "logging.h"

// #define AVTHREAD_VERBOSE 1

AVThread::AVThread(const char* name, FrameFilter& outfilter, FrameFifoContext fifo_ctx) 
    : DecoderThread(name, outfilter, fifo_ctx) 
    {
    }


AVThread::~AVThread() {
}

