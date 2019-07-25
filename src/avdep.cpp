/*
 * avdep.cpp : Helper functions for calling FFmpeg API functions directly from Valkka
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
 *  @file    avdep.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.13.0 
 *  
 *  @brief 
 *
 *  @section DESCRIPTION
 *
 */ 

#include "avdep.h"

void ffmpeg_av_register_all() {
 av_register_all();
}

void ffmpeg_av_log_set_level(int level) {
  av_log_set_level(level);
}

