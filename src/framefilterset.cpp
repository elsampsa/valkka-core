/*
 * framefilterset.cpp : Classes using several framefilters
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
 *  @file    framefilterset.cpp
 *  @author  Sampsa Riikonen
 *  @date    2018
 *  @version 1.3.5 
 *  
 *  @brief   Classes using several framefilters
 */ 

#include "framefilterset.h"

Switch::Switch(const char* name, FrameFilter* next) : 
  name(name), 
  channel0( (std::string(name)+std::string("_channel0")).c_str(), next ), 
  channel1( (std::string(name)+std::string("_channel1")).c_str(), next ),
  n_channel(-1) {
}


Switch::~Switch() {
}
  

void Switch::setChannel(int i) {
  if (i<0 or i>1) {
    std::cout << "Switch: setChannel: wrong channel " << i << std::endl;
  }
  else { // eh.. using std::vector would make more sense .. or doing this at the python level if there are many channels
    if (n_channel>=0) { // close previous channel
      if (n_channel==0) { // 0 was open, close it
        channel0.unSet();
      }
      else if (n_channel==1) { // 1 was open, close it
        channel1.unSet();
      }
    }
    // set and open the new channel
    n_channel=i;
    if (n_channel==0) { // 0 was open, close it
      channel0.set();
    }
    else if (n_channel==1) { // 1 was open, close it
      channel1.set();
    }
  }
}

int Switch::getCurrentChannel() {
  return n_channel;
}


FrameFilter* Switch::getInputChannel(int i) {
  if (i<0 or i>1) {
    std::cout << "Switch: getInputChannel: wrong channel " << i << std::endl;
  }
  else {
    if (i==0) { // 0 was open, close it
      return &channel0;
    }
    else if (i==1) { // 1 was open, close it
      return &channel1;
    }
  }
}



DoubleGate::DoubleGate(const char* name, FrameFilter* next, FrameFilter* next2) : 
  name(name), 
  channel0( (std::string(name)+std::string("_channel0")).c_str(), next ), 
  channel1( (std::string(name)+std::string("_channel1")).c_str(), next2),
  n_channel(-1) {
}


DoubleGate::~DoubleGate() {
}


void DoubleGate::setChannel(int i) {
  if (i<0 or i>1) {
    std::cout << "DoubleGate: setChannel: wrong channel " << i << std::endl;
  }
  else { // eh.. using std::vector would make more sense .. or doing this at the python level if there are many channels
    if (n_channel>=0) { // close previous channel
      if (n_channel==0) { // 0 was open, close it
        channel0.unSet();
      }
      else if (n_channel==1) { // 1 was open, close it
        channel1.unSet();
      }
    }
    // set and open the new channel
    n_channel=i;
    if (n_channel==0) { // 0 was open, close it
      channel0.set();
    }
    else if (n_channel==1) { // 1 was open, close it
      channel1.set();
    }
  }
}

int DoubleGate::getCurrentChannel() {
  return n_channel;
}


FrameFilter* DoubleGate::getInputChannel(int i) {
  if (i<0 or i>1) {
    std::cout << "DoubleGate: getInputChannel: wrong channel " << i << std::endl;
  }
  else {
    if (i==0) { // 0 was open, close it
      return &channel0;
    }
    else if (i==1) { // 1 was open, close it
      return &channel1;
    }
  }
}




