/*
 * chains.cpp : Some ready-made chains with filters, decoders and framefifos
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
 *  @file    chains.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.0 
 *  
 *  @brief Some ready-made chains with filters, decoders and framefifos
 */ 

#include "chains.h"

BasicChain::BasicChain(LiveConnectionType contype, const char* adr, SlotNumber n_slot, FifoFrameFilter& gl_in_filter) : 
  contype(contype),
  adr(adr), 
  n_slot(n_slot),
  gl_in_filter(gl_in_filter)
  {
    std::string framefifoname = std::string("av_fifo")          +std::to_string(n_slot);
    std::string avthreadname  = std::string("avthread")         +std::to_string(n_slot);
    std::string fffname       = std::string("av_in_filter")     +std::to_string(n_slot);
    std::string iffname       = std::string("live_out_filter")  +std::to_string(n_slot);
    
    av_fifo        =new FrameFifo(framefifoname.c_str(),10);
    avthread       =new AVThread(avthreadname.c_str(),*av_fifo,gl_in_filter); // , 3); // thread affinity
    av_in_filter   =new FifoFrameFilter(fffname.c_str(),*av_fifo);
    // live_out_filter=new InfoFrameFilter(iffname.c_str(),av_in_filter);
    live_out_filter=new DummyFrameFilter(iffname.c_str(),true,av_in_filter);
    
    // ctx = (LiveConnectionContext){contype, std::string(adr), n_slot, live_out_filter};
    // ctx = new LiveConnectionContext(contype, std::string(adr), n_slot, live_out_filter);
    ctx.connection_type=contype; ctx.address=std::string(adr); ctx.slot=n_slot; ctx.framefilter=live_out_filter;
    render_ctx=0;
    window_id =0;
  }


BasicChain::~BasicChain() {
  delete av_fifo;
  delete avthread;
  delete av_in_filter;
  delete live_out_filter;
  // delete ctx;
}

