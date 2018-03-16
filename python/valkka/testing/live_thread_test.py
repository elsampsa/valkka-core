"""
live_thread_test.py : Cloning some of the stuff in "live_thread_test.cpp"

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

@file    live_thread_test.py
@author  Sampsa Riikonen
@date    2017
@version 0.3.5 
  
@brief Cloning some of the stuff in "live_thread_test.cpp"
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
  print(name,":","** @@Starting, playing and stopping a single rtsp connection **")

  if (stream_1==None):
    print("ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1")
    return
  print(name,"** test rtsp stream 1: ",stream_1)
  
  """
  filtergraph:
  (LiveThread:livethread) --> {FrameFilter:dummyfilter)
  """
  livethread  =LiveThread("livethread")
  dummyfilter =DummyFrameFilter("dummy",True)
  
  ctx=LiveConnectionContext()
  ctx.slot=1
  ctx.connection_type=LiveConnectionType_rtsp
  ctx.address=stream_1
  ctx.framefilter=dummyfilter
  
  print("starting live thread")
  livethread.startCall()
  
  time.sleep(2.0)
  
  livethread.registerStreamCall(ctx)
    
  time.sleep(1.0)
  
  livethread.playStreamCall(ctx)  
  time.sleep(3.0)

  livethread.stopStreamCall(ctx)
  
  print("stopping live thread")
  livethread.stopCall()

  
def main():
  """
  test1()
  return
  """
  if (len(sys.argv)<2):
    print("Needs test number")
  else:
    sname="test"+str(sys.argv[1])+"()"
    try:
      exec(sname)
    except Exception as e:
      print("Could not execute",sname,":",str(e))
  
    
main()

  
