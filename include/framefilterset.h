#ifndef framefilterset_HEADER_GUARD
#define framefilterset_HEADER_GUARD
/*
 * framefilterset.h : Classes using several framefilters
 * 
 * Copyright 2018 Valkka Security Ltd. and Sampsa Riikonen.
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
 *  @file    framefilterset.h
 *  @author  Sampsa Riikonen
 *  @date    2018
 *  @version 0.17.5 
 *  
 *  @brief   Classes using several framefilters
 */ 


#include "framefilter.h"

/** Switch between two input streams
 * 
 * The idea is this:
 * 
 * \verbatim
 *  filter_chain_0 ----> +--------+
 *                       |        |
 *                       | Switch | -----> output_filter_chain
 *                       |        |
 *  filter_chain_1 ----> +--------+
 * \endverbatim
 * 
 * - The start filter of output_filter_chain is given as a parameter to Switch constructor
 * - Input FrameFilter pointers are obtained by calling Switch::getInputChannel(i), where i=0 or 1.
 * - Switch chooses between active input chains by calling setChannel(i), where i=0 or 1
 * 
 * @ingroup filters_tag
 */
class Switch {                                       // <pyapi>
  
public:                                              // <pyapi>
  /** Default ctor
   * 
   * @param name  Name of the switch
   * @param next  Output FrameFilter
   * 
   */
  Switch(const char* name, FrameFilter* next=NULL);  // <pyapi>
  virtual ~Switch();                                 // <pyapi>
  
protected:
  std::string name;
  CachingGateFrameFilter channel0; ///< Input framefilter for channel 0
  CachingGateFrameFilter channel1; ///< Input framefilter for channel 1
  int n_channel;            ///< Current active channel
  
public:                                             // <pyapi>
  void setChannel(int i);                           // <pyapi>
  int  getCurrentChannel();                         // <pyapi>
  FrameFilter* getInputChannel(int i);              // <pyapi>
};                                                  // <pyapi>



/** Gates between two input streams
 * 
 * The idea is this:
 * 
 * \verbatim
 *  filter_chain_0 ----> +--------+ ----> output_filter_chain0
 *                       | Double |
 *                       | Gate   | 
 *                       |        |
 *  filter_chain_1 ----> +--------+ ----> output_filter_chain1
 * \endverbatim
 * 
 * - Only one output_filter_chain is active at a time
 * - Start filters of output_filter_chains are given as parameters to DoubleGate constructor
 * - Input FrameFilter pointers are obtained by calling Switch::getInputChannel(i), where i=0 or 1.
 * - Switch chooses between chains by calling setChannel(i), where i=0 or 1
 * 
 * @ingroup filters_tag
 */
class DoubleGate {                                   // <pyapi>
  
public:                                              // <pyapi>
  DoubleGate(const char* name, FrameFilter* next=NULL, FrameFilter* next2=NULL);  // <pyapi>
  virtual ~DoubleGate();                             // <pyapi>
  
protected:
  std::string name;
  CachingGateFrameFilter channel0;
  CachingGateFrameFilter channel1;
  int n_channel;
  
public:                                             // <pyapi>
  void setChannel(int i);                           // <pyapi>
  int  getCurrentChannel();                         // <pyapi>
  FrameFilter* getInputChannel(int i);              // <pyapi>
};                                                  // <pyapi>


#endif


