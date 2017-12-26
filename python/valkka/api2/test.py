"""
test.py : Some api level 2 tests.  Look at valkka-examples git repo for more.

Copyright 2017 Valkka Security Ltd. and Sampsa Riikonen.

Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>

This file is part of Valkka library.

Valkka is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Valkka is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Valkka.  If not, see <http://www.gnu.org/licenses/>. 

@file    NAME.py
@author  Sampsa Riikonen
@date    2017
@version 0.1
  
@brief Some api level 2 tests.  Look at valkka-examples git repo for more.
"""

from valkka.api2.threads import LiveThread, OpenGLThread, ShmemClient
from valkka.api2.chains import ShmemFilterchain



def test1():
  st="""Empty test
  """
  pre=pre_mod+"test1 :"
  print(pre,st)
  
  livethread=LiveThread(
    name="live_thread"
    )
  
  openglthread=OpenGLThread(
    name    ="mythread",
    n1440p  =5,
    verbose =True
    )
  

def test2():
  st="""Empty test
  """
  pre=pre_mod+"test2 :"
  print(pre,"st")
  

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
