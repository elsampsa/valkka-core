/*
 * tools.h : Auxiliary routines
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
 *  @file    tools.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.2.0 
 *  
 *  @brief Auxiliary routines
 *
 */ 


#include "common.h"
#include "sizes.h"
#include "logging.h"

int64_t NANOSEC_PER_SEC = 1000000000;

long int getCurrentMsTimestamp(); ///< Utility function: returns current unix epoch timestamp in milliseconds

long int getMsDiff(timeval tv1, timeval tv2); ///< Utility function: return timedif of two timeval structs in milliseconds

bool slotOk(SlotNumber n_slot); ///< Checks the slot number range

void normalize_timespec(struct timespec *ts, time_t sec, int64_t nanosec);
