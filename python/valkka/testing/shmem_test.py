"""
shmem_test.py : Clone of the client side test of "shmem_test.cpp"

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

@file    shmem_test.py
@author  Sampsa Riikonen
@date    2017
@version 0.3.6 
  
@brief Clone of the client side test of "shmem_test.cpp" : launch "shmem_test 3 0" from the cpp side and after that, this python3 program
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

if (stream_1==None):
    print("ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1")
    return
  print(name,"** test rtsp stream 1: ",stream_1)

if (stream_2==None):
    print("ERROR: missing test stream 2: set environment variable VALKKA_TEST_RTSP_2")
    return
  print(name,"** test rtsp stream 2: ",stream_2)

if (stream_sdp==None):
    print("ERROR: missing test sdp stream: set environment variable VALKKA_TEST_SDP")
    return
  print(name,"** test sdp stream: ",stream_sdp)

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
  """
  first start the cpp test program (SharedMemRingBuffer server side) with './shmem_test 3 0'
  then start this program (SharedMemRingBuffer python client side) with 'python3 shmem_test.py 1'
  """
  name ="@PYTEST: "+__file__+" : "+inspect.stack()[0][3]
  print(name,":","** @@Python client side of SharedMemRingBuffer **")
  
  index_p =new_intp()
  isize_p =new_intp()
  
  rb=SharedMemRingBuffer("testing",10,30*1024*1024,False) # shmem ring buffer on the client side
  
  shmem_list=[]
  for i in range(10):
    shmem_list.append(getNumpyShmem(rb,i)) # if you're looking for this, its defined in the .i swig interface file.  :)
  
  while(True):
    i=int(input("number of buffers to read. 0 exits>"))
    if (i<1): break
    while(i>0):
      rb.clientPull(index_p, isize_p);
      index=intp_value(index_p); isize=intp_value(isize_p)
      print("Current index, size=",index,isize)
      print("Payload=",shmem_list[index][0:min(isize,10)])
      i-=1
      
      
def test2():
  name ="@PYTEST: "+__file__+" : "+inspect.stack()[0][3]
  print(name,":","** @@Python test of ShmemFrameFilter **")
  shmem           =SharedMemFrameFilter("test", 10, 1024*1024*30);
  live_out_filter =InfoFrameFilter("live_out_filter",shmem);
  livethread      =LiveThread("livethread");  
      
  
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

  
