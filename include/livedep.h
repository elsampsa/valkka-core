/*
 * livedep.h : A list/recompilation of common header files for live555
 * 
 * Copyright 2017 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Valkka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Valkka.  If not, see <http://www.gnu.org/licenses/>. 
 * 
 */

/** 
 *  @file    livedep.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1.0 
 *  
 *  @brief List of common header files
 *
 */

// live555 header files
// for custom installation : copy/link header files to "$valkka/include/ext/", place relevant "*.so" files to "$valkka/lib" (see the ln_live.bash scripts)
#include "UsageEnvironment/UsageEnvironment.hh"
#include "BasicUsageEnvironment/BasicUsageEnvironment0.hh"
#include "BasicUsageEnvironment/BasicUsageEnvironment.hh"

#include "groupsock/NetAddress.hh"
#include "groupsock/GroupsockHelper.hh"

#include "liveMedia/liveMedia.hh"
#include "liveMedia/Media.hh"
#include "liveMedia/MediaSession.hh"
#include "liveMedia/RTSPClient.hh"
#include "liveMedia/FramedSource.hh"
#include "liveMedia/H264VideoRTPSource.hh"

#include "liveMedia/RTSPClient.hh" 
