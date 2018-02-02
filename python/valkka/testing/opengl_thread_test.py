"""
opengl_thread_test.py : Cloning some of the stuff in "opengl_thread_test.cpp"

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

@file    opengl_thread_test.py
@author  Sampsa Riikonen
@date    2017
@version 0.3.0 
  
@brief Cloning some of the stuff in "opengl_thread_test.cpp"
"""

import sys
import os
import inspect
import time
from valkka.valkka_core import *

"""
# some broilerplate for you sir:
if (VERSION_MAJOR<0)
if (VERSION_MINOR<0)
if (stream_1==None)
if (stream_2==None)
if (stream_sdp==None)
"""


try:
  stream_1 =os.environ["VALKKA_TEST_RTSP_1"]
except:
  stream_1 =None
try:
  stream_2 =os.environ["VALKKA_TEST_RTSP_2"]
except:
  stream_2 =None
try:
  stream_sdp =os.environ["VALKKA_TEST_SDP"]
except:
  stream_sdp =None
  

def test2():
  name ="@PYTEST: "+__file__+" : "+inspect.stack()[0][3]
  print(name,":","** @@OpenGLThread live rendering **")
  
  if (stream_1==None):
    print("ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1")
    return
  print(name,"** test rtsp stream 1: ",stream_1)
  
  """
  filtergraph:
  (LiveThread:livethread) --> {InfoFrameFilter:live_out_filter} --> {FifoFrameFilter:av_in_filter} --> [FrameFifo:av_fifo] -->> (AVThread:avthread) --> {FifoFrameFilter:gl_in_gilter} --> 
  --> [OpenGLFrameFifo:gl_fifo] -->> (OpenGLThread:glthread)
  """
  glthread        =OpenGLThread       ("glthread",
                                       10,     # n720p
                                       10,     # n1080p
                                       0,      # n1440p
                                       0,      # n4K
                                       10,     # naudio
                                       100,    # msbuftime
                                       -1      # thread affinity
                                       )
  
  gl_fifo         =glthread.getFifo();# get gl_fifo from glthread
  gl_in_filter    =FifoFrameFilter    ("gl_in_filter",   gl_fifo)
  
  av_fifo         =FrameFifo          ("av_fifo",10)                 
  avthread        =AVThread           ("avthread",       av_fifo,gl_in_filter)  #[av_fifo] -->> (avthread) --> {gl_in_filter}
  
  av_in_filter    =FifoFrameFilter    ("av_in_filter",   av_fifo)
  live_out_filter =InfoFrameFilter    ("live_out_filter",av_in_filter)
  livethread      =LiveThread         ("livethread")
  
  ctx=LiveConnectionContext()
  ctx.slot=1
  ctx.connection_type=LiveConnectionType_rtsp
  ctx.address=stream_1
  ctx.framefilter=live_out_filter
  
  print(name,"starting threads")
  glthread.startCall() # start running OpenGLThread!
  
  window_id=glthread.createWindow()
  glthread.makeCurrent(window_id)
  print("new x window",window_id)
  
  livethread.startCall()
  avthread.  startCall()

  avthread.decodingOnCall()
    
  print(name,"registering stream")
  livethread.registerStreamCall(ctx)
  
  print(name,"playing stream !")
  livethread.playStreamCall(ctx)
  
  # (1)
  glthread.newRenderGroupCall(window_id);
  time.sleep(1.0)
  
  i=glthread.newRenderContextCall(1, window_id, 0)
  print("got render context id",i)
  time.sleep(1.0)
  
  glthread.delRenderContextCall(i)
  ok=glthread.delRenderGroupCall(window_id)
  
  print(name,"stopping threads")
  livethread.stopCall()
  avthread.  stopCall()
  glthread.  stopCall()
  print(name,"All threads stopped")
  time.sleep(1.0)
  print(name,"Leaving context")
  
  
def main():
  test2()
  return
  
  if (len(sys.argv)<2):
    print("Needs test number")
  else:
    sname="test"+str(sys.argv[1])+"()"
    try:
      exec(sname)
    except Exception as e:
      print("Could not execute",sname,":",str(e))
  
    
main()

  
