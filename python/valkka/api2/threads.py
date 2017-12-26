"""
threads.py : api level 1 => api level 2 encapsulation for LiveThread and OpenGLThread

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

@file    threads.py
@author  Sampsa Riikonen
@date    2017
@version 0.1
  
@brief api level 1 => api level 2 encapsulation for LiveThread and OpenGLThread
"""

from multiprocessing import Process, Pipe, Event
import select
import time
from valkka import valkka_core
from valkka.api2.tools import *

pre_mod="valkka.api2 : "


def safe_select(l1,l2,l3,timeout=0):
  """
  print("safe_select: enter")
  select.select(l1,l2,l3,0)
  print("safe_select: return")
  return True
  """
  try:
    if (timeout>0):
      select.select(l1,l2,l3,timeout)
    else:
      select.select(l1,l2,l3)
    # print("safe_select: ping")
  except (select.error, v):
    if (v[0] != errno.EINTR): 
      print("select : ERROR READING SOCKET!")
      raise
      return False # dont read socket ..
    else:
      # print("select : giving problems..")
      return False # dont read socket
  else:
    # print("select : returning true!")
    return True # read socket
  # print("select : ?")


class LiveThread:
  # LiveThread(const char* name, int core_id=-1);
  
  parameter_defs={
    "name"      : (str,"live_thread"),
    "affinity"  : (int,-1),
    "verbose"   : (bool, False)
    }
  

  def __init__(self,**kwargs):
    self.pre=self.__class__.__name__+" : " # auxiliary string for debugging output
    parameterInitCheck(LiveThread.parameter_defs,kwargs,self) # checks kwargs agains parameter_defs, attach ok'd parameters to this object as attributes
    
    # this is the "api-level 1" object.  Just a swig-wrapped cpp instance.
    self.core=valkka_core.LiveThread(self.name, self.affinity)
    
    self.active=True
    self.core.startCall()
    
  
  def registerStream(self,ctx):
    self.core.registerStreamCall(ctx)
  
  
  def deregisterStream(self,ctx):
    self.core.deregisterStreamCall(ctx)
    
  
  def playStream(self,ctx):
    self.core.playStreamCall(ctx)
  
  
  def stopStream(self,ctx):
    self.core.stopStreamCall(ctx)
  
  
  def close(self):
    if (self.active):
      if (self.verbose): print(self.pre,"stopping core.LiveThread")
      self.core.stopCall()
      self.active=False
      
  # """
  def __del__(self):
    self.close()
  # """


class OpenGLThread:
  # OpenGLThread(const char* name, unsigned short n720p=0, unsigned short n1080p=0, unsigned short n1440p=0, unsigned short n4K=0, unsigned short naudio=0, unsigned msbuftime=100, int core_id=-1);

  parameter_defs={
    "name"      : (str,"live_thread"),
    "n720p"     : (int, 10),
    "n1080p"    : (int, 10),
    "n1440p"    : (int, 0),
    "n4K"       : (int, 0),
    "naudio"    : (int, 10),
    "msbuftime" : (int, 100),   
    "affinity"  : (int, -1),
    "verbose"   : (bool,False)
    }

  
  def __init__(self,**kwargs):
    self.pre=self.__class__.__name__+" : " # auxiliary string for debugging output
    parameterInitCheck(self.parameter_defs,kwargs,self) # check kwargs agains parameter_defs, attach ok'd parameters to this object as attributes
    
    # this is the "api-level 1" object.  Just a swig-wrapped cpp instance.
    self.core=valkka_core.OpenGLThread(self.name, self.n720p, self.n1080p, self.n1440p, self.n4K, self.naudio, self.msbuftime, self.affinity)
    
    self.render_groups =[] # list of registered x windowses
    self.render_counts =[] # how many slots are mapped per each x window
    self.tokens        =[] # unique tokens identifying the slot => x window mappings
    self.token_to_group=[] # gives group corresponding to each token by token index
    
    self.active=True
    self.core.startCall()


  def close(self):
    if (self.active):
      self.core.stopCall()
      self.active=False

      
  def __del__(self):
    self.close()
    

  def createWindow(self):
    """Create an x window.  For debugging/testing only.  Returns x-window id.
    """
    window_id=self.core.createWindow()
    self.core.makeCurrent(window_id)
    return window_id


  def connect(self,slot=0,window_id=0):
    """Returns a unique token identifying the slot => x-window mapping.  Returns 0 if mapping failed.
    """
    if (slot==0):
      raise(AssertionError("valid slot number missing"))
    
    if (window_id==0):
      raise(AssertionError("valid window_id missing"))
    
    if (window_id not in self.render_groups):
      if (self.verbose):
        print(self.pre,"connect : new render group :",window_id)
      ok=self.core.newRenderGroupCall(window_id)
      if (ok):
        self.render_groups.append(window_id)
        self.render_counts.append(0)
      else:
        print(self.pre,"connect : WARNING : creating render group failed :",window_id)
        return 0
      
    i=self.render_groups.index(window_id)
    self.render_counts[i]+=1
      
    token=self.core.newRenderContextCall(slot, window_id, 0) # slot, window_id, z (not functional at the moment)
    if (token==0):
      print(self.pre,"connect : WARNING : creating render contex failed")
      # TODO: check if the render group should be eliminated..?
      return 0
    self.tokens.append(token)
    self.token_to_group.append(window_id)
    return token
    
      
  def disconnect(self,token=0):
    if (token==0):
      raise(AssertionError("invalid token 0"))
    
    if (token not in self.tokens):
      print(self.pre,"disconnect : WARNING : no such token :",token)
    else:
      if (self.verbose):
        print(self.pre,"disconnect : removing token :",token)
      ok=self.core.delRenderContextCall(token)
      if (not ok):
        print(self.pre,"disconnect : WARNING : could not remove token :",token)
      
      # get the index
      i        =self.tokens.index(token)
      window_id=self.token_to_group[i]
      
      # remove token from the books
      self.tokens.pop(i)
      self.token_to_group.pop(i)
      
      i=self.render_groups.index(window_id)
      self.render_counts[i]-=1
      if (self.render_counts[i]<=0): # nothing to map into this x window anymore
        if (self.verbose):
          print(self.pre,"disconnect : removing render group :",window_id)
        ok=self.core.delRenderGroupCall(window_id)
        if (not ok):
          print(self.pre,"disconnect : WARNING : could not remove render group :",window_id)
        self.render_groups.pop(i) # remove from the books
    
    
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
  
  
class ValkkaRGBProcess(Process):
  
  
  signal_defs={ # each key corresponds to a backend method
    "test_"    : {"test_int": int, "test_str": str},
    "stop_"    : []
    }
  
  parameter_defs={
    "image_dimensions"       : (tuple,(1920//4,1080//4)),
    "n_ringbuffer"           : (int,10),                 # size of the ringbuffer
    "name"                   : str,
    "mstimeout"              : (int,1000),
    "verbose"                : (bool,False)
    }
  
  
  def __init__(self,**kwargs):
    super().__init__()
    self.pre=self.__class__.__name__+" : " # auxiliary string for debugging output
    parameterInitCheck(ValkkaRGBProcess.parameter_defs,kwargs,self) # check kwargs agains parameter_defs, attach ok'd parameters to this object as attributes
    typeCheck(self.image_dimensions[0],int)
    typeCheck(self.image_dimensions[1],int)
    self.n_bytes=self.image_dimensions[0]*self.image_dimensions[1]*3
    
    self.signal_in            = Event()
    self.signal_out           = Event()
    self.pipe, self.childpipe = Pipe() # communications pipe.  Frontend uses self.pipe, backend self.childpipe
  
    self.signal_in.clear()
    self.signal_out.clear()
    
    
  
  def preRun(self):
    """After the fork, but before starting the process loop
    """
    self.client=ShmemClient(
      name        =self.name,
      n_ringbuffer=self.n_ringbuffer,
      n_bytes     =self.n_bytes,
      mstimeout   =self.mstimeout,
      verbose     =self.verbose
      )

    
  def postRun(self):
    """Just before process exit
    """
    print(self.pre,"post: bye!")
  
  
  def run(self):
    """After the fork
    """
    self.preRun()
    self.running=True
    
    while(self.running):
      index, isize = self.client.pull()
      if (index==None): # semaphore timed out
        print("Semaphore timeout")
      else:
        print("Current index, size=",index,isize)
        print("Payload=",self.client.shmem_list[index][0:min(isize,10)])

      if (self.signal_in.is_set()):
        signal_dic=self.childpipe.recv()
        method_name=signal_dic.pop("name")
        method=getattr(self,method_name)
        method(**signal_dic)
        self.signal_in.clear()
        self.signal_out.set()
        

    self.postRun()
    
    
    
  # ** backend methods **
  def test_(self,test_int=0,test_str="nada"):
    print(self.pre,"test_ signal received with",test_int,test_str)
    
  
  def stop_(self):
    self.running=False
    
    
    
  # ** frontend methods: these communicate with the backend via pipes **
  
  def sendSignal(self,**kwargs): # sendSignal(name="test",test_int=1,test_str="kokkelis")
    try:
      name=kwargs.pop("name")
    except KeyError:
      raise(AttributeError("Signal name missing"))
    
    model=self.signal_defs[name] # a dictionary: {"parameter_name" : parameter_type}
    
    for key in kwargs:
      model_type      =model[key] # raises error if user is using undefined signal
      parameter_type  =kwargs[key].__class__
      if (model_type==parameter_type):
        pass
      else:
        raise(AttributeError("Wrong type for parameter "+str(key)))
      
    kwargs["name"]=name
      
    self.pipe.send(kwargs)
    self.signal_out.clear()
    self.signal_in. set()  # indicate that there is a signal
    self.signal_out.wait() # wait for the backend to clear the signal 
      
      
  def stop(self):
    self.sendSignal(name="stop_")
  
  
  def test(self,**kwargs):
    dictionaryCheck(self.signal_defs["test_"],kwargs)
    kwargs["name"]="test_"
    self.sendSignal(**kwargs)
    
  
  
  

def test1():
  st="""Test LiveThread encapsulation
  """
  import time
  
  pre=pre_mod+"test1 :"
  print(pre,st)
  
  livethread=LiveThread(
    name="live_thread"
    )
  
  print(pre,"sleepin'")
  time.sleep(3)
  print(pre,"bye!")
  
  # livethread.close() # TODO: should thread destructor stop the thread if not already stopped?



def test2():
  st="""Test OpenGLThread encapsulation
  """
  pre=pre_mod+"test2 :"
  print(pre,st)
  
  openglthread=OpenGLThread(
    name    ="mythread",
    n1440p  =5,
    verbose =True
    )
  
  win_id  =openglthread.createWindow()
  win_id1 =openglthread.createWindow()
  win_id2 =openglthread.createWindow()
  
  token=openglthread.connect(1,win_id)
  print(pre,"got token",token)
  openglthread.disconnect(token)
  print(pre,"disconnected token",token)
  
  print(pre,"connect token")
  token1  =openglthread.connect(1,win_id1)
  print(pre,"connect token")
  token11 =openglthread.connect(2,win_id1)
  print(pre,"connect token")
  token2  =openglthread.connect(3,win_id2)
  
  print(pre,"disconnect token",token1)
  openglthread.disconnect(token1)
  print(pre,"disconnect AGAIN token",token1)
  openglthread.disconnect(token1)
  print(pre,"disconnect token",token11)
  openglthread.disconnect(token11)
  print(pre,"disconnect token",token2)
  openglthread.disconnect(token2)
  
  # openglthread.close()


def test3():
  st="""Test OpenGLThread encapsulation again
  """
  pre=pre_mod+"test3 :"
  print(pre,st)
  
  openglthread=OpenGLThread(
    name    ="mythread",
    n1440p  =5,
    verbose =True
    )


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


def test5():
  st="""Test ValkkaRGBProcess
  """
  pre=pre_mod+"test4 :"
  print(pre,st)
  
  p=ValkkaRGBProcess(
    image_dimensions = (1920//4,1080//4),
    n_ringbuffer     = 10,
    name             = "testing",
    mstimeout        = 1000,
    verbose          = True
  )
  
  p.start()
  time.sleep(5)
  p.test(test_int=10,test_str="kikkelis")
  time.sleep(3)
  p.stop()
  
  
  
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



