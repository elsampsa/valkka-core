"""
basic1.py : Some more custom filterchain classes for different use cases

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

@file    basic1.py
@author  Sampsa Riikonen
@date    2018
@version 0.13.2 
  
@brief Some more custom filterchain classes for different use cases
"""

import sys
import time
import random
from valkka import core # so, everything that has .core, refers to the api1 level (i.e. swig wrapped cpp code)
from valkka.api2.threads import LiveThread, OpenGLThread, FileThread # api2 versions of the thread classes
from valkka.api2.tools import parameterInitCheck, typeCheck
pre_mod="valkka.api2.chains.basic1 : "


class BasicFilterchain1:
  """This class implements the following filterchain:
  
  ::
    
    (LiveThread:livethread) -->> (AVThread:avthread) -->> (OpenGLThread:glthread)
  
  i.e. the stream is decoded by an AVThread and sent to the OpenGLThread for presentation
  
  LiveConnectionContext and FileContext are managed by the user
  
  Different to BasicFilterChain: has a file context
  """
  
  parameter_defs={
    "openglthread" : OpenGLThread,
    "slot"         : int,
    
    # these are for the AVThread instance:
    "n_basic"      : (int,20), # number of payload frames in the stack
    "n_setup"      : (int,20), # number of setup frames in the stack
    "n_signal"     : (int,20), # number of signal frames in the stack
    "flush_when_full" : (bool, False), # clear fifo at overflow
    
    "affinity"     : (int,-1),
    "msreconnect"  : (int,0),
    "verbose"      : (bool,False)
    }
  
  
  def __init__(self, **kwargs):
    self.pre=self.__class__.__name__+" : " # auxiliary string for debugging output
    parameterInitCheck(self.parameter_defs,kwargs,self) # check for input parameters, attach them to this instance as attributes
    self.init()
    
    
  def init(self):
    self.idst=str(id(self))
    self.makeChain()
    self.startThreads()
    self.active=True
    
    
  def __del__(self):
    self.close()
    
    
  def close(self):
    if (self.active):
      if (self.verbose):
        print(self.pre,"Closing threads and contexes")
      self.decodingOff()
      self.stopThreads()
      self.active=False
    

  def makeChain(self):
    """Create the filter chain
    """    
    self.gl_in_filter  =self.openglthread.getInput() # get input FrameFilter from OpenGLThread
    
    self.framefifo_ctx=core.FrameFifoContext()
    self.framefifo_ctx.n_basic           =self.n_basic
    self.framefifo_ctx.n_setup           =self.n_setup
    self.framefifo_ctx.n_signal          =self.n_signal
    self.framefifo_ctx.flush_when_full   =self.flush_when_full
    
    self.avthread      =core.AVThread("avthread_"+self.idst, self.gl_in_filter, self.framefifo_ctx)
    self.avthread.setAffinity(self.affinity)
    self.av_in_filter  =self.avthread.getFrameFilter() # get input FrameFilter from AVThread

    

  def setLiveContext(self,address):
    """
    The user is responsible for:
    
    self.livethread.registerStream(self.ctx)
    self.livethread.playStream(self.ctx)
    self.livethread.stopStream(self.ctx)
    self.livethread.deregisterStream(self.ctx)
    """
    
    self.live_ctx=core.LiveConnectionContext()
    self.live_ctx.slot=self.slot                          # slot number identifies the stream source
    
    if (address.find("rtsp://")==0):
      self.live_ctx.connection_type=core.LiveConnectionType_rtsp
    else:
      self.live_ctx.connection_type=core.LiveConnectionType_sdp # this is an rtsp connection
    
    self.live_ctx.address=address         
    # stream address, i.e. "rtsp://.."
    
    self.live_ctx.framefilter=self.av_in_filter
    
    self.live_ctx.msreconnect=self.msreconnect # if no frames received in this time, attempt to reconnect
    
    
  def setFileContext(self,filename):
    """
    The user is responsible for:
    
    filethread.openFileStreamCall(self.file_ctx);
    file_ctx.seektime_=10000;
    filethread.seekFileStreamCall(self.file_ctx);
    filethread.playFileStreamCall(self.file_ctx);
    filethread.stopFileStreamCall(self.file_ctx);
    """
    
    self.file_ctx =core.FileContext()
    """ # filecontext from the cpp headers:
    std::string    filename;        ///< incoming: the filename                                      // <pyapi>
    SlotNumber     slot;            ///< incoming: a unique stream slot that identifies this stream  // <pyapi>
    FrameFilter*   framefilter;     ///< incoming: the frames are feeded into this FrameFilter       // <pyapi>
    long int       seektime_;       ///< incoming: used by signal seek_stream                        // <pyapi>
    long int*      duration;        ///< outgoing: duration of the stream                            // <pyapi>
    long int*      mstimestamp;     ///< outgoing: current position of the stream (stream time)      // <pyapi>
    FileState      status;          ///< outgoing: status of the file                                // <pyapi>
    """
    self.file_ctx.filename       =filename
    self.file_ctx.slot           =self.slot
    self.file_ctx.framefilter    =self.av_in_filter
    self.file_ctx.seektime_      =0
    # self.file_ctx.duration       
    # self.file_ctx.mstimestamp   
    # self.file_ctx.status         
    
    
  def fileStatusOk(self):
    return (self.file_ctx.status!=core.FileState_error)
    
    
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
      
      
      
class ShmemFilterchain1(BasicFilterchain1):
  """A filter chain with a shared mem hook
  
  ::
  
    (LiveThread:livethread) -->> (AVThread:avthread) --+ 
                                                       |   main branch
    {ForkFrameFilter: fork_filter} <-------------------+
               |
      branch 1 +-->> (OpenGLThread:glthread)
               |
      branch 2 +--> {IntervalFrameFilter: interval_filter} --> {SwScaleFrameFilter: sws_filter} --> {RGBShmemFrameFilter: shmem_filter}
  
  
  * Frames are decoded in the main branch from H264 => YUV
  * The stream of YUV frames is forked into two branches
  * branch 1 goes to OpenGLThread that interpolates YUV to RGB on the GPU
  * branch 2 goes to interval_filter that passes a YUV frame only once every second.  From there, frames are interpolated on the CPU from YUV to RGB and finally passed through shared memory to another process.
  
  LiveConnectionContext and FileContext are managed by the user
  """
  
  parameter_defs={ # additional parameters to the mother class
    "shmem_image_dimensions"  : (tuple,(1920//4,1080//4)), # images passed over shmem are full-hd/4 reso
    "shmem_image_interval"    : (int,1000),              # .. passed every 1000 milliseconds
    "shmem_ringbuffer_size"   : (int,10),                # size of the ringbuffer
    "shmem_name"              : None
    }
  
  parameter_defs.update(BasicFilterchain1.parameter_defs) # don't forget!
  
  
  def __init__(self, **kwargs):
    self.pre=self.__class__.__name__+" : " # auxiliary string for debugging output
    parameterInitCheck(self.parameter_defs,kwargs,self) # check for input parameters, attach them to this instance as attributes
    typeCheck(self.shmem_image_dimensions[0],int)
    typeCheck(self.shmem_image_dimensions[1],int)
    self.init()
            
            
  def makeChain(self):
    """Create the filter chain
    """
    if (self.shmem_name==None):
      self.shmem_name ="shmemff"+self.idst
    # print(self.pre,self.shmem_name)
    
    # self.n_bytes =self.shmem_image_dimensions[0]*self.shmem_image_dimensions[1]*3
    n_buf   =self.shmem_ringbuffer_size
    
    # branch 1
    self.gl_in_filter    =self.openglthread.getInput() # get input FrameFilter from OpenGLThread
    
    # branch 2
    # print(self.pre,"using shmem name",self.shmem_name)
    # print(self.shmem_name)
    self.shmem_filter    =core.RGBShmemFrameFilter  (self.shmem_name,  n_buf, self.shmem_image_dimensions[0], self.shmem_image_dimensions[1]) # shmem id, cells, width, height 
    # self.shmem_filter    =core.InfoFrameFilter        ("info"+self.idst) # debug
    
    self.sws_filter      =core.SwScaleFrameFilter      ("sws_filter"+self.idst, self.shmem_image_dimensions[0], self.shmem_image_dimensions[1], self.shmem_filter)
    self.interval_filter =core.TimeIntervalFrameFilter ("interval_filter"+self.idst, self.shmem_image_interval, self.sws_filter)
    
    # fork: writes to branches 1 and 2
    # self.fork_filter     =core.ForkFrameFilter         ("fork_filter"+self.idst,self.gl_in_filter,self.sws_filter) # FIX
    self.fork_filter     =core.ForkFrameFilter         ("fork_filter"+self.idst,self.gl_in_filter,self.interval_filter)
    
    # main branch
    self.framefifo_ctx=core.FrameFifoContext()
    self.framefifo_ctx.n_basic           =self.n_basic
    self.framefifo_ctx.n_setup           =self.n_setup
    self.framefifo_ctx.n_signal          =self.n_signal
    self.framefifo_ctx.flush_when_full   =self.flush_when_full
    
    self.avthread      =core.AVThread("avthread_"+self.idst, self.fork_filter, self.framefifo_ctx) # AVThread writes to self.fork_filter
    self.avthread.setAffinity(self.affinity)
    self.av_in_filter  =self.avthread.getFrameFilter() # get input FrameFilter from AVThread
    # self.av_in_filter is used by BasicFilterchain.createContext that passes self.av_in_filter to LiveThread
    # # self.live_out_filter =core.InfoFrameFilter    ("live_out_filter"+self.idst,self.av_in_filter)


  def getShmemPars(self):
    """Returns shared mem name that should be used in the client process and the ringbuffer size
    """
    # SharedMemRingBuffer(const char* name, int n_cells, std::size_t n_bytes, int mstimeout=0, bool is_server=false); // <pyapi>
    # return self.shmem_name, self.shmem_ringbuffer_size, self.n_bytes
    return self.shmem_name, self.shmem_ringbuffer_size, self.shmem_image_dimensions
      

def test1():
  st=""" Test single live stream
  """
  pre=pre_mod+"test1 :"
  print(pre,st)
  
  livethread=LiveThread(
    name   ="live_thread",
    verbose=True
    )
  
  openglthread=OpenGLThread(
    name    ="mythread",
    n_1440p  =5,
    verbose =True
    )
  
  # now livethread and openglthread are running ..
  
  chain=BasicFilterchain1(
    openglthread=openglthread,
    slot=1
    )

  chain.setLiveContext("rtsp://admin:nordic12345@192.168.1.41")
  
  livethread.registerStream   (chain.live_ctx)
  livethread.playStream       (chain.live_ctx)
  print("sleeping for some secs")
  time.sleep(5)
  livethread.stopStream       (chain.live_ctx)
  livethread.deregisterStream (chain.live_ctx)
  print("bye!")


def test2():
  st=""" Test single file stream
  """
  pre=pre_mod+"test2 :"
  print(pre,st)
  
  filethread=FileThread(
    name   ="filethread",
    verbose=True
    )
  
  openglthread=OpenGLThread(
    name    ="mythread",
    n_1440p  =5,
    verbose =True
    )
  
  # now livethread and openglthread are running ..
  
  chain=BasicFilterchain1(
    openglthread=openglthread,
    slot=1
    )

  chain.setFileContext("kokkelis.mkv")
  
  filethread.openStream(chain.file_ctx)
  # chain.file_ctx.seektime_=10000
  chain.file_ctx.seektime_=0
  filethread.seekStream(chain.file_ctx)
  filethread.playStream(chain.file_ctx)
  filethread.stopStream(chain.file_ctx)
  
  print("sleeping for some secs")
  time.sleep(3)
  print("bye!")



def test3():
  st=""" Test ShmemFilterchain
  """
  pre=pre_mod+"test3 :"
  print(pre,st)
  
  livethread=LiveThread(
    name   ="live_thread",
    verbose=True
    )
  
  openglthread=OpenGLThread(
    name    ="mythread",
    n_1440p  =5,
    verbose =True
    )
  
  # now livethread and openglthread are running ..
  
  chain=ShmemFilterchain1(
    openglthread=openglthread,
    slot=1,
    shmem_image_dimensions=(1920//4,1080//4),  # images passed over shmem are full-hd/4 reso
    shmem_image_interval  =1000,              # .. passed every 1000 milliseconds
    shmem_ringbuffer_size =10                 # size of the ringbuffer
    )


  chain.setLiveContext("rtsp://admin:nordic12345@192.168.1.41")
  
  livethread.registerStream   (chain.live_ctx)
  livethread.playStream       (chain.live_ctx)
  print("sleeping for some secs")
  time.sleep(5)
  livethread.stopStream       (chain.live_ctx)
  livethread.deregisterStream (chain.live_ctx)

  print("bye!")


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
