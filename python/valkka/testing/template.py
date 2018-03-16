"""
NAME.py :

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

@file    NAME.py
@author  Sampsa Riikonen
@date    2017
@version 0.3.5 
  
@brief 

@section DESCRIPTION
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
  name ="@PYTEST: "+__file__+" : "+inspect.stack()[0][3]
  print(name,":","** @@Test description **")
  
  
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

  
