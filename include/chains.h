/*
 * chains.h : Some ready-made chains with filters, decoders and framefifos
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
 *  @file    chains.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
 *  
 *  @brief Some ready-made chains with filters, decoders and framefifos
 */ 

#include "queues.h"
#include "livethread.h"
#include "avthread.h"
#include "filters.h"

/** This class implements the following filtergraph:
 * --> {InfoFrameFilter:live_out_filter} --> {FifoFrameFilter:av_in_filter} --> [FrameFifo:av_fifo] -->> (AVThread:avthread) --> {FifoFrameFilter:gl_in_gilter} -->
 * 
 * The idea is that you can create vectors/lists of this class for multiple streams.  The members are pointers, because many of them are non-copiable (for example, FrameFifo and AVThread).  
 * 
 * The best idea is to implement a filterchain class like this in the python level, and not in cpp as done here.  This is just an example class and mainly for cpp debugging purposes.
 * 
 */
class BasicChain {
  
public:
  BasicChain(LiveConnectionType contype, const char* adr, SlotNumber n_slot, FifoFrameFilter& gl_in_filter);
  ~BasicChain();
  
protected: // initialized at constructor init list
  LiveConnectionType contype;
  std::string        adr;
  SlotNumber         n_slot;
  FifoFrameFilter&   gl_in_filter;
  FrameFifo*         av_fifo;     //    ("av_fifo",10);                 
  FifoFrameFilter*   av_in_filter;  //  ("av_in_filter",av_fifo);
  // InfoFrameFilter*   live_out_filter; // ("live_out_filter",&av_in_filter);
  DummyFrameFilter*   live_out_filter; // ("live_out_filter",&av_in_filter);
  
public: // initialized at constructor init list
  AVThread*          avthread;    //    ("avthread",av_fifo,gl_in_filter);  // [av_fifo] -->> (avthread) --> {gl_in_filter}
  
protected: // initialized at constructor
  LiveConnectionContext ctx;
  
protected: // aux variables
  int    render_ctx;
  Window window_id;
  
public: // getters & setters
  LiveConnectionContext& getCtx()            {return ctx;}
  SlotNumber             getSlot()           {return n_slot;}
  void                   setRenderCtx(int i) {render_ctx=i;}
  int                    getRenderCtx()      {return render_ctx;}
  void                   setWindow(Window i) {window_id=i;}
  Window                 getWindow()         {return window_id;}
};

