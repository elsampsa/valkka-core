/*
 * avdep.cpp : Helper functions for calling FFmpeg API functions directly from Valkka
 * 
 * Copyright 2017 Sampsa Riikonen and Petri Eranko.
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Valkka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Valkka.  If not, see <http://www.gnu.org/licenses/>. 
 * 
 */

/** 
 *  @file    avdep.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
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

void ffmpeg_av_log_set_level(unsigned int level) {
  av_log_set_level(level);
}

