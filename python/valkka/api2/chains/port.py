"""
port.py : Port classes for managed filterchains (see "managed.py").  "Ports" are terminals of the filterchains that require resources (say, decoding or connection to a certain x screen)

 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>
 *
 */

@file    manage.py
@author  Sampsa Riikonen
@date    2018
@version 1.5.1 
  
@brief   Port classes for managed filterchains (see "managed.py").  "Ports" are terminals of the filterchains that require resources (say, decoding or connection to a certain x screen)
"""

import sys
import time
import random
from valkka import core # so, everything that has .core, refers to the api1 level (i.e. swig wrapped cpp code)
from valkka.api2.threads import LiveThread, OpenGLThread # api2 versions of the thread classes
from valkka.api2.tools import parameterInitCheck, typeCheck
pre_mod="valkka.api2.chains.port : " 


class BitmapPort:
  """Generic Port class.  A "port" is something that requests bitmap stream (can be analyzer, video-on-screen, etc.)
  """
  
  def __init__(self):
    pass
  
  
  def getWidth(self):
    """Returns the width (in pixels) this port is requesting
    """
    raise(AssertionError(self.pre,"Port: getWidth : virtual method"))
  
  
  def getHeight(self):
    """Returns the height (in pixels) this port is requesting
    """
    raise(AssertionError(self.pre,"Port: getHeight : virtual method"))
  
  

class ViewPort(BitmapPort):
  """A view of a bitmap somewhere on the screen (and on a certain x-screen)
  """
  
  parameter_defs={
    "window_id"       : (int,0),
    "x_screen_num"    : (int,0)
    }
  
  def __init__(self, **kwargs):
    self.pre=self.__class__.__name__+" : " # auxiliary string for debugging output
    parameterInitCheck(self.parameter_defs,kwargs,self) # check for input parameters, attach them to this instance as attributes
  
  
  def getWindowId(self):
    return self.window_id


  def setWindowId(self,id):
    self.window_id=id

  
  def getXScreenNum(self):
    return self.x_screen_num

  def setXScreenNum(self,n):
    self.x_screen_num=n



  
def test1():
  
  class NewViewPort(ViewPort):
    pass
  
  print(issubclass(NewViewPort,ViewPort))
  
  
  
    
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



