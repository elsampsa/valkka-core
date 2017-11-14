/*
 * tools.cpp : Auxiliary routines
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
 *  @file    tools.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
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


bool slotOk(SlotNumber n_slot) {
  if (n_slot>I_MAX_SLOTS) {
    std::cout << "WARNING! slot overflow with "<<n_slot<<" increase I_MAX_SLOTS in sizes.h"<<std::endl;
    return false;
  }
  return true;
}

