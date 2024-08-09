#ifndef AVDEP_HEADER_GUARD 
#define AVDEP_HEADER_GUARD

/*
 * avdep.h : A list/recompilation of common ffmpeg/libav header files
 * 
 * Copyright 2017-2023 Valkka Security Ltd. and Sampsa Riikonen
 * Copyright 2024 Sampsa Riikonen
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
 *  @file    avdep.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.6.1 
 *  
 *  @brief List of common ffmpeg/libav header files. Definition of some functions to call FFmpeg API directly from Valkka
 * 
 */

// ffmpeg header files
// for custom installation : copy/link header files to "$valkka/include/ext/", place relevant "*.so" files to "$valkka/lib" (see the ln_ffmpeg.bash scripts)
extern "C" { // realizing this took me half a day : https://stackoverflow.com/questions/15625468/libav-linking-error-undefined-references
#include <libavcodec/avcodec.h>
// #include <libavcodec/vdpau.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
// #include <libavutil/parseutils.h>
#include <libavutil/log.h>
#include <libswscale/swscale.h>
}
// libavcodec libavformat libavutil libswscale


// When linking againts Valkka, we don't want to link "manually" agains all dependent libraries .. for this reason, we can't (and shouldn't call their symbols directly)
// We don't want the user to interact or even see the live555 and ffmpeg apis
// For special cases where we need that, use these helper functions:

void ffmpeg_av_register_all();           // <pyapi>
void ffmpeg_av_log_set_level(int level); // <pyapi>

#endif
