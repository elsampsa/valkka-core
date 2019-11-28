#ifndef TOOLS_HEADER_GUARD 
#define TOOLS_HEADER_GUARD

/*
 * tools.h : Auxiliary routines
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
 *  @file    tools.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.14.1 
 *  
 *  @brief Auxiliary routines
 *
 */ 


#include "common.h"
#include "constant.h"
#include "logging.h"

/** posix time doodle:
*
* time_t (sec)
* 
* struct timeval (tv_sec, tv_usec)
* 
* struct timespec (tv_sec, tv_nsec)
* 
*/

static const int64_t NANOSEC_PER_SEC = 1000000000;

long int getCurrentMsTimestamp(); ///< Utility function: returns current unix epoch timestamp in milliseconds.  Uses timeval

long int getMsDiff(timeval tv1, timeval tv2); ///< Utility function: return timedif of two timeval structs in milliseconds

struct timeval msToTimeval(long int mstimestamp); ///< Milliseconds to timeval

long int timevalToMs(struct timeval time); /// Timeval to milliseconds

bool slotOk(SlotNumber n_slot); ///< Checks the slot number range

void normalize_timespec(struct timespec *ts, time_t sec, int64_t nanosec);


#ifdef BIG_ENDIAN
uint32_t deserialize_uint32_big_endian(unsigned char *buffer)
{
    uint32_t value = 0;

    value |= buffer[0] << 24;
    value |= buffer[1] << 16;
    value |= buffer[2] << 8;
    value |= buffer[3];
    return value;
}
#else // either not defined or little endian
// deserialize value from big endian in little endian system
// byte1 byte2 byte3 byte4 => byte4 byte3 ..
uint32_t deserialize_uint32_big_endian(unsigned char *buffer)
{
    uint32_t value = 0;

    value |= buffer[3] << 24;
    value |= buffer[2] << 16;
    value |= buffer[1] << 8;
    value |= buffer[0];
    return value;
}
#endif


#endif
