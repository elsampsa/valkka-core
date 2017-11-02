"""
av_live_thread_test.py : Cloning some of the stuff in "av_live_thread_test.cpp"

Copyright 2017 Valkka Security Ltd. and Sampsa Riikonen.

Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>

This file is part of Valkka library.

Valkka is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Valkka is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Valkka.  If not, see <http://www.gnu.org/licenses/>. 

@file    av_live_thread_test.py
@author  Sampsa Riikonen
@date    2017
@version 0.1
  
@brief Cloning some of the stuff in "av_live_thread_test.cpp"
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
  

def test1():
  name ="@PYTEST: "+__file__+" : "+inspect.stack()[0][3]
  print(name,":","** @@Send frames from live to av thread **")

  if (stream_1==None):
    print("ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1")
    return
  print(name,"** test rtsp stream 1: ",stream_1)
  
  """ 
  filtergraph:
  (LiveThread:livethread) --> {FrameFilter:info} --> {FifoFrameFilter:terminal} --> [FrameFifo:in_fifo] -->> (AVThread:avthread) --> {InfoFrameFilter:decoded_info}
  """
  
  decoded_info   =InfoFrameFilter("decoded_info");

  in_fifo        =FrameFifo      ("in_fifo", 10);                
  avthread       =AVThread       ("avthread",in_fifo,decoded_info);
  
  terminal       =FifoFrameFilter("terminal",in_fifo); 
  info           =InfoFrameFilter("info",    terminal); 
  livethread     =LiveThread     ("livethread"); 
  
  ctx=LiveConnectionContext()
  ctx.slot=1
  ctx.connection_type=LiveConnectionType_rtsp
  ctx.address=stream_1
  ctx.framefilter=info
  
  verbose=True
  
  print(name,"starting threads")
  livethread.startCall()
  avthread.  startCall()

  avthread.decodingOnCall()
  
  time.sleep(2.0)
  
  print(name,"registering stream")
  livethread.registerStreamCall(ctx)
  
  time.sleep(1.0)
  
  print(name,"playing stream !")
  livethread.playStreamCall(ctx)
  
  time.sleep(3.0)
  
  print(name,"stopping threads")
  livethread.stopCall()
  avthread.  stopCall()
  
  
def main():
  if (len(sys.argv)<2):
    print("Needs test number")
  else:
    sname="test"+str(sys.argv[1])+"()"
    try:
      exec(sname)
    except Exception as e:
      print("Could not execute",sname,":",str(e))
  
    
main()

  
