"""
basic2.py : Some basic classes encapsulating filter chains.  User must define the endpoints of the filterchains.

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

@file    basic.py
@author  Sampsa Riikonen
@date    2017
@version 0.17.0 
  
@brief Some basic classes encapsulating filter chains.  User must define the endpoints of the filterchains
"""

import sys
import time
import random
from valkka import core # so, everything that has .core, refers to the api1 level (i.e. swig wrapped cpp code)
from valkka.api2.threads import LiveThread, OpenGLThread # api2 versions of the thread classes
from valkka.api2.tools import parameterInitCheck, typeCheck
pre_mod="valkka.api2.chains.basic2 : "


class OpenFilterchain:
  """This class implements the following filterchain:
  
  :: 
                                                                                           ...
                                                                                          |
    (LiveThread:livethread) -->> (AVThread:avthread) --> {ForkFrameFilterN:forkfilter} ---+  request forked streams with method "connect"
                                                                                          |...
  
  """
  
  parameter_defs={
    "livethread"       : LiveThread,
    "address"          : str,
    "slot"             : int,
    
    # these are for the AVThread instance:
    "n_basic"      : (int,20), # number of payload frames in the stack
    "n_setup"      : (int,20), # number of setup frames in the stack
    "n_signal"     : (int,20), # number of signal frames in the stack
    "flush_when_full" : (bool, False), # clear fifo at overflow
    
    "affinity"     : (int,-1),
    "verbose"      : (bool,False),
    "msreconnect"  : (int,0),
    
    "time_correction"   : None,    # Timestamp correction type: TimeCorrectionType_none, TimeCorrectionType_dummy, or TimeCorrectionType_smart (default)
    "recv_buffer_size"  : (int,0), # Operating system socket ringbuffer size in bytes # 0 means default
    "reordering_mstime" : (int,0)  # Reordering buffer time for Live555 packets in MILLIseconds # 0 means default
    }
  
  
  def __init__(self, **kwargs):
    self.pre=self.__class__.__name__+" : " # auxiliary string for debugging output
    parameterInitCheck(self.parameter_defs,kwargs,self) # check for input parameters, attach them to this instance as attributes
    self.init()
    
    
  def init(self):
    self.idst=str(id(self))
    self.makeChain()
    self.createContext()
    self.startThreads()
    self.active=True
    
    
  def __del__(self):
    self.close()
    
    
  def close(self):
    if (self.active):
      if (self.verbose):
        print(self.pre,"Closing threads and contexes")
      self.decodingOff()
      self.closeContext()
      self.stopThreads()
      self.active=False
    
    
  def makeChain(self):
    """Create the filter chain
    """
    self.fork_filter=core.ForkFrameFilterN("av_fork_at_slot_"+str(self.slot)) # FrameFilter chains can attached to ForkFrameFilterN after it's been instantiated
    
    self.framefifo_ctx=core.FrameFifoContext()
    self.framefifo_ctx.n_basic           =self.n_basic
    self.framefifo_ctx.n_setup           =self.n_setup
    self.framefifo_ctx.n_signal          =self.n_signal
    self.framefifo_ctx.flush_when_full   =self.flush_when_full
    
    self.avthread      =core.AVThread("avthread_"+self.idst, self.fork_filter, self.framefifo_ctx)
    self.avthread.setAffinity(self.affinity)
    self.av_in_filter  =self.avthread.getFrameFilter() # get input FrameFilter from AVThread

  
  def connect(self,name,framefilter):
    return self.fork_filter.connect(name,framefilter)
    
    
  def disconnect(self,name):
    return self.fork_filter.disconnect(name)
    

  def createContext(self):
    """Creates a LiveConnectionContext and registers it to LiveThread
    """
    # define stream source, how the stream is passed on, etc.
    
    self.ctx=core.LiveConnectionContext()
    self.ctx.slot=self.slot                          # slot number identifies the stream source
    
    if (self.address.find("rtsp://")==0):
      self.ctx.connection_type=core.LiveConnectionType_rtsp
    else:
      self.ctx.connection_type=core.LiveConnectionType_sdp # this is an rtsp connection
    
    self.ctx.address=self.address         
    # stream address, i.e. "rtsp://.."
    
    self.ctx.framefilter=self.av_in_filter
    
    self.ctx.msreconnect=self.msreconnect
    
    # some extra parameters
    """
    // ctx.time_correction =TimeCorrectionType::none;
    // ctx.time_correction =TimeCorrectionType::dummy;
    // default time correction is smart
    // ctx.recv_buffer_size=1024*1024*2;  // Operating system ringbuffer size for incoming socket
    // ctx.reordering_time =100000;       // Live555 packet reordering treshold time (microsecs)
    """
    if (self.time_correction!=None): self.ctx.time_correction =self.time_correction
    self.ctx.recv_buffer_size =self.recv_buffer_size
    self.ctx.reordering_time  =self.reordering_mstime*1000 # from millisecs to microsecs
    
    # send the information about the stream to LiveThread
    self.livethread.registerStream(self.ctx)
    self.livethread.playStream(self.ctx)

      
  def closeContext(self):
    self.livethread.stopStream(self.ctx)
    self.livethread.deregisterStream(self.ctx)
    
      
  def startThreads(self):
    """Starts thread required by the filter chain
    """
    self.avthread.startCall()


  def stopThreads(self):
    """Stops threads in the filter chain
    """
    self.avthread.stopCall()
    

  def decodingOff(self):
    self.avthread.decodingOffCall()


  def decodingOn(self):
    self.avthread.decodingOnCall()

      
def main():
  pre=pre_mod+"main :"
  print(pre,"main: arguments: ",sys.argv)
  if (len(sys.argv)<2):
    print(pre,"main: needs test number")
  else:
    st="test"+str(sys.argv[1])+"()"
    exec(st)
  
  
if (__name__=="__main__"):
  main()
