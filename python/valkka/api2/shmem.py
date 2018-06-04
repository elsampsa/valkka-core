"""
shmem.py : Encapsulation for Valkka's cpp shared memory client

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

@file    shmem.py
@author  Sampsa Riikonen
@date    2017
@version 0.4.6 
  
@brief   Encapsulation for Valkka's cpp shared memory client
"""
from valkka import valkka_core
from valkka.api2.tools import *

pre_mod="valkka.api2.shmem: "


class ShmemClient:
  """A shared memory ringbuffer client.  The idea is here, that the ringbuffer "server" is instantiated at the cpp side.  Client must have exactly the same name (that identifies the shmem segments), and the number of ringbuffer elements.
  
  :param name:               name identifying the shared mem segment.  Must be same as in the server side.
  :param n_ringbuffer:       Number of elements in the ringbuffer.  Must be same as in the server side.
  :param n_bytes:            Size of each element in the ringbuffer.
  :param mstimeout:          Timeout for semaphores.  Default=0=no timeout.
  :param verbose:            Be verbose or not.  Default=False.
  
  """
  
  parameter_defs={
    "name"              : str,         # :param name:               name identifying the shared mem segment.  Must be same as in the server side.
    "n_ringbuffer"      : (int,10),    # :param n_ringbuffer:       Number of elements in the ringbuffer.  Must be same as in the server side.
    "n_bytes"           : int,         # :param n_bytes:            Size of each element in the ringbuffer.
    "mstimeout"         : (int,0),     # :param mstimeout:          Timeout for semaphores.  Default=0=no timeout.
    "verbose"           : (bool,False) # :param verbose:            Be verbose or not.  Default=False.
  }
  
  
  def __init__(self,**kwargs):
    self.pre=self.__class__.__name__+" : " # auxiliary string for debugging output
    parameterInitCheck(ShmemClient.parameter_defs,kwargs,self) # check kwargs agains parameter_defs, attach ok'd parameters to this object as attributes

    self.index_p =valkka_core.new_intp()
    self.isize_p =valkka_core.new_intp()
  
    # print(self.pre,"shmem name=",self.name)
    self.core=valkka_core.SharedMemRingBuffer(self.name,self.n_ringbuffer,self.n_bytes,self.mstimeout,False) # shmem ring buffer on the client side
  
    self.shmem_list=[]
    for i in range(self.n_ringbuffer):
      self.shmem_list.append(valkka_core.getNumpyShmem(self.core,i)) # if you're looking for this, it's defined in the .i swig interface file.  :)
  
    
  def pull(self):
    """If semaphore was timed out (i.e. nothing was written to the ringbuffer) in mstimeout milliseconds, returns: None, None.  Otherwise returns the index of the shmem segment and the size of data written.
    """
    got=self.core.clientPull(self.index_p, self.isize_p)
    index=valkka_core.intp_value(self.index_p); isize=valkka_core.intp_value(self.isize_p)
    if (self.verbose): print(self.pre,"current index, size=",index,isize)
    if (got):
      return index, isize
    else:
      return None, None
  
  
  
class ShmemRGBClient:
  """A shared memory ringbuffer client for RGB images.  The idea is here, that the ringbuffer "server" is instantiated at the cpp side.  Client must have exactly the same name (that identifies the shmem segments), and the number of ringbuffer elements.
  
  :param name:               name identifying the shared mem segment.  Must be same as in the server side.
  :param n_ringbuffer:       Number of elements in the ringbuffer.  Must be same as in the server side.
  :param width:              RGB image width
  :param height:             RGb image height
  :param mstimeout:          Timeout for semaphores.  Default=0=no timeout.
  :param verbose:            Be verbose or not.  Default=False.
  
  """
  
  parameter_defs={
    "name"              : str,         # :param name:               name identifying the shared mem segment.  Must be same as in the server side.
    "n_ringbuffer"      : (int,10),    # :param n_ringbuffer:       Number of elements in the ringbuffer.  Must be same as in the server side.
    "width"             : int,         # :param width:              RGB image width
    "height"            : int,         # :param height:             RGb image height
    "mstimeout"         : (int,0),     # :param mstimeout:          Timeout for semaphores.  Default=0=no timeout.
    "verbose"           : (bool,False) # :param verbose:            Be verbose or not.  Default=False.
  }
  
  
  def __init__(self,**kwargs):
    self.pre=self.__class__.__name__+" : " # auxiliary string for debugging output
    parameterInitCheck(ShmemRGBClient.parameter_defs,kwargs,self) # check kwargs agains parameter_defs, attach ok'd parameters to this object as attributes

    self.index_p =valkka_core.new_intp()
    self.isize_p =valkka_core.new_intp()
  
    # print(self.pre,"shmem name=",self.name)
    self.core=valkka_core.SharedMemRingBufferRGB(self.name,self.n_ringbuffer,self.width,self.height,self.mstimeout,False) # shmem ring buffer on the client side
  
    self.shmem_list=[]
    for i in range(self.n_ringbuffer):
      self.shmem_list.append(valkka_core.getNumpyShmem(self.core,i)) # if you're looking for this, it's defined in the .i swig interface file.  :)
  
    
  def pull(self):
    """If semaphore was timed out (i.e. nothing was written to the ringbuffer) in mstimeout milliseconds, returns: None, None.  Otherwise returns the index of the shmem segment and the size of data written.
    """
    got=self.core.clientPull(self.index_p, self.isize_p)
    index=valkka_core.intp_value(self.index_p); isize=valkka_core.intp_value(self.isize_p)
    if (self.verbose): print(self.pre,"current index, size=",index,isize)
    if (got):
      return index, isize
    else:
      return None, None
  
  

def test4():
  st="""Test shmem client.  First start the cpp test program (SharedMemRingBuffer server side) with './shmem_test 3 0'
  """
  # at cpp side:
  # SharedMemRingBuffer rb("testing",10,30*1024*1024,1000,true); // name, ncells, bytes per cell, timeout, server or not
  pre=pre_mod+"test4 :"
  print(pre,st)
  
  client=ShmemClient(
    name        ="testing",
    n_ringbuffer=10,
    n_bytes     =30*1024*1024,
    mstimeout   =0,
    verbose     =True
    )

  while(True):
    i=int(input("number of buffers to read. 0 exits>"))
    if (i<1): break
    while(i>0):
      index, isize = client.pull()
      print("Current index, size=",index,isize)
      print("Payload=",client.shmem_list[index][0:min(isize,10)])
      i-=1

  

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
