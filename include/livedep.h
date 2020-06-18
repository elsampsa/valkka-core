/*
 * livedep.h : A list/recompilation of common header files for live555
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
 *  @file    livedep.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.17.5 
 *  
 *  @brief List of common header files
 *
 */

// live555 header files
// for custom installation : copy/link header files to "$valkka/include/ext/", place relevant "*.so" files to "$valkka/lib" (see the ln_live.bash scripts)
#include "UsageEnvironment.hh"
#include "BasicUsageEnvironment0.hh"
#include "BasicUsageEnvironment.hh"

#include "NetAddress.hh"
#include "GroupsockHelper.hh"

#include "liveMedia.hh"
#include "Media.hh"
#include "MediaSession.hh"
#include "RTSPClient.hh"
#include "FramedSource.hh"
#include "H264VideoRTPSource.hh"
#include "RTSPClient.hh" 
