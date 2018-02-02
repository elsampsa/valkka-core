/*
 * tools.cpp : Auxiliary routines
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
 *  @file    tools.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.0 
 *  
 *  @brief Auxiliary routines
 *
 *  @section DESCRIPTION
 *  
 *
 */ 

#include "tools.h"

long int getCurrentMsTimestamp() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return time.tv_sec*1000+time.tv_usec/1000;
}


long int getMsDiff(timeval tv1, timeval tv2) {
  return (tv1.tv_sec-tv2.tv_sec)*1000 + (tv1.tv_usec-tv2.tv_usec)/1000;
}


struct timeval msToTimeval(long int mstimestamp) {
  struct timeval fPresentationTime;
  fPresentationTime.tv_sec   =(mstimestamp/1000); // secs
  fPresentationTime.tv_usec  =(mstimestamp-fPresentationTime.tv_sec*1000)*1000; // microsecs
  return fPresentationTime;
}


bool slotOk(SlotNumber n_slot) {
  if (n_slot>I_MAX_SLOTS) {
    std::cout << "WARNING! slot overflow with "<<n_slot<<" increase I_MAX_SLOTS in sizes.h"<<std::endl;
    return false;
  }
  return true;
}


// normalized_timespec : normalize to nanosec and sec
void normalize_timespec(struct timespec *ts, time_t sec, int64_t nanosec)
{
  while (nanosec >= NANOSEC_PER_SEC) {
    asm("" : "+rm"(nanosec));
    nanosec -= NANOSEC_PER_SEC;
    ++sec;
  }
  while (nanosec < 0) {
    asm("" : "+rm"(nanosec));
    nanosec += NANOSEC_PER_SEC;
    --sec;
  }
  ts->tv_sec  = sec;
  ts->tv_nsec = nanosec;
}


