#ifndef threadsignal_HEADER_GUARD
#define threadsignal_HEADER_GUARD
/*
 * signal.h : Signals used by thread classes
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
 *  @file    signal.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.4.7 
 *  
 *  @brief   Signals used by thread classes
 */ 

#include "constant.h"


/** Different signal types understood by Threads (sending them to the Thread by an interrupt (Thread::sendSignal) or in the Frame stream)
 */
enum class SignalType {
  av,
  gl
};

/** Signals used by AVThread
 * 
 * @ingroup decoding_tag
 */
enum class AVSignal {
  none,
  exit,
  on,  ///< turn decoding on 
  off  ///< turn decoding off
};

  
/** Redefinition of characteristic signal contexts (info that goes with the signal) for AVThread thread
 * 
 * @ingroup decoding_tag
*/
struct AVSignalContext {
  AVSignal signal;
  // AVConnectionContext connection_context; // in the case we want pass more information
};


/** Signal information for OpenGLThread
 * 
 * @ingroup openglthread_tag
 */
struct OpenGLSignalPars  {   // used by signals:                                
  SlotNumber    n_slot;        ///< in: new_render_context                                         
  Window        x_window_id;   ///< in: new_render_context, new_render_group, del_render_group     
  unsigned int  z;             ///< in: new_render_context                                         
  int           render_ctx;    ///< in: del_render_context, out: new_render_context                
  bool          success;       ///< return value: was the call succesful?                          
};

// std::ostream &operator<<(std::ostream &os, OpenGLSignalPars const &m);
std::ostream &operator<<(std::ostream &os, OpenGLSignalPars const &m) {
 return os << "<OpenGLSignalPars: slot="<<m.n_slot<<" x_window_id="<<m.x_window_id<<" z="<<m.z<<" render_context="<<m.render_ctx<<" success="<<m.success<<">";
};

/** Signals used by OpenGLThread
 * 
 * @ingroup openglthread_tag
 */
enum class OpenGLSignal {
  none,                 ///< null signal
  exit,                 ///< exit
  info,                 ///< used by API infoCall
  new_render_group,     ///< used by API newRenderCroupCall
  del_render_group,     ///< used by API delRenderGroupCall
  new_render_context,   ///< used by API newRenderContextCall
  del_render_context    ///< used by API delRenderContextCall
};

/** Encapsulates data sent by the signal
  * 
  * Has the enumerated signal from OpenGLThread::Signals class plus any other necessary data, represented by OpenGLSignalContext.
  * 
  * @ingroup openglthread_tag
  */
struct OpenGLSignalContext {
  OpenGLSignal      signal;   ///< The signal
  // OpenGLSignalPars  *pars;     ///< Why pointers? .. we have return values here // nopes .. not anymore
  OpenGLSignalPars  pars;
};


#endif
