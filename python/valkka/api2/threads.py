"""
threads.py : api level 1 => api level 2 encapsulation for LiveThread and OpenGLThread

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

@file    threads.py
@author  Sampsa Riikonen
@date    2017
@version 0.3.0 
  
@brief api level 1 => api level 2 encapsulation for LiveThread and OpenGLThread
"""

from multiprocessing import Process, Pipe, Event
import select
import time
from valkka import valkka_core
from valkka.api2.tools import *

valkka_core.XInitThreads()

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
      res=select.select(l1,l2,l3,timeout)
    else:
      res=select.select(l1,l2,l3)
    # print("safe_select: ping")
  except (select.error, v):
    if (v[0] != errno.EINTR): 
      print("select : ERROR READING SOCKET!")
      raise
      return [[],[],[]] # dont read socket ..
    else:
      # print("select : giving problems..")
      return [[],[],[]] # dont read socket
  else:
    # print("select : returning true!")
    return res # read socket
  # print("select : ?")



class Namespace: # generic namespace ..
    
  def __init__(self):
    pass



class LiveThread:
  # LiveThread(const char* name, int core_id=-1);
  
  parameter_defs={
    "name"      : (str,"live_thread"),
    "n_stack"   : (int,0), # stack for the incoming frames
    "affinity"  : (int,-1),
    "verbose"   : (bool, False)
    }
  

  def __init__(self,**kwargs):
    self.pre=self.__class__.__name__+" : " # auxiliary string for debugging output
    parameterInitCheck(LiveThread.parameter_defs,kwargs,self) # checks kwargs agains parameter_defs, attach ok'd parameters to this object as attributes
    
    # this is the "api-level 1" object.  Just a swig-wrapped cpp instance.
    self.core=valkka_core.LiveThread(self.name, self.n_stack, self.affinity)
    
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



class FileThread:
  # FileThread(const char* name, int core_id=-1);
  
  parameter_defs={
    "name"      : (str,"file_thread"),
    "affinity"  : (int,-1),
    "verbose"   : (bool, False)
    }
  

  def __init__(self,**kwargs):
    self.pre=self.__class__.__name__+" : " # auxiliary string for debugging output
    parameterInitCheck(FileThread.parameter_defs,kwargs,self) # checks kwargs agains parameter_defs, attach ok'd parameters to this object as attributes
    
    # this is the "api-level 1" object.  Just a swig-wrapped cpp instance.
    self.core=valkka_core.FileThread(self.name, self.affinity)
    
    self.active=True
    self.core.startCall()
    
  """
  void closeFileStreamCall      (FileContext &file_ctx); ///< API method: registers a stream                                // <pyapi> 
  void openFileStreamCall       (FileContext &file_ctx); ///< API method: de-registers a stream                             // <pyapi>
  void seekFileStreamCall       (FileContext &file_ctx); ///< API method: seek to a certain point                           // <pyapi>
  void playFileStreamCall       (FileContext &file_ctx); ///< API method: starts playing the stream and feeding frames      // <pyapi>
  void stopFileStreamCall       (FileContext &file_ctx); ///< API method: stops playing the stream and feeding frames       // <pyapi>
  """
  
  def openStream(self,ctx):
    self.core.openFileStreamCall(ctx)
  
  
  def closeStream(self,ctx):
    self.core.closeFileStreamCall(ctx)
  
  
  def seekStream(self,ctx):
    self.core.seekFileStreamCall(ctx)
  
  
  def playStream(self,ctx):
    self.core.playFileStreamCall(ctx)
  

  def stopStream(self,ctx):
    self.core.stopFileStreamCall(ctx)
  
  
  def close(self):
    if (self.active):
      if (self.verbose): print(self.pre,"stopping core.FileThread")
      self.core.stopCall()
      self.active=False
      
  # """
  def __del__(self):
    self.close()
  # """




class OpenGLThread:
  # OpenGLThread(const char* name, unsigned short n720p=0, unsigned short n1080p=0, unsigned short n1440p=0, unsigned short n4K=0, unsigned msbuftime=100, int core_id=-1);

  parameter_defs={
    "name"      : (str,"live_thread"),
    "n720p"     : (int, 10),
    "n1080p"    : (int, 10),
    "n1440p"    : (int, 0),
    "n4K"       : (int, 0),
    # "naudio"    : (int, 10),
    "msbuftime" : (int, 100),   
    "affinity"  : (int, -1),
    "verbose"   : (bool,False)
    }

  
  def __init__(self,**kwargs):
    self.pre=self.__class__.__name__+" : " # auxiliary string for debugging output
    parameterInitCheck(self.parameter_defs,kwargs,self) # check kwargs agains parameter_defs, attach ok'd parameters to this object as attributes
    
    # this is the "api-level 1" object.  Just a swig-wrapped cpp instance.
    self.core=valkka_core.OpenGLThread(self.name, self.n720p, self.n1080p, self.n1440p, self.n4K, self.msbuftime, self.affinity)
    
    self.render_groups =[] # list of registered x windowses
    self.render_counts =[] # how many slots are mapped per each x window
    self.tokens        =[] # unique tokens identifying the slot => x window mappings
    self.token_to_group=[] # gives group corresponding to each token by token index
    
    self.active=True
    self.core.startCall()


  def close(self):
      #if (self.active):
      if (self.verbose): print(self.pre,"stopping core.OpenGLThread")
      self.core.stopCall()
      self.active=False

      
  def __del__(self):
    self.close()
    

  def createWindow(self,show=True):
    """Create an x window.  For debugging/testing only.  Returns x-window id.
    """
    window_id=self.core.createWindow(show)
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
    
    
  def hadVsync(self):
    return self.core.getVsyncAtStartup()>0
    
  
    
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
  
  
  
class ValkkaProcess(Process):
  """
  Frontend: the part of the forked process that keeps running in the curren virtual memory space
  Backend : the part of the forked process that runs in its own virtual memory space
  
  This class has frontend methods, that you should only call from frontend, and backend methods that are called only from the backend
  
  To avoid confusion, backend methods are designated with "_", except for the "run()" method, that's always in the backend
  
  Frontend methods use a pipe to send a signal to backend that then handles the signal with a backend method having the same name (but with "_" in the end)
  
  Backend methods can, in a similar fashion, send signals to the frontend using a pipe.  In frontend, a listening thread is needed.   That thread can then call
  the handleSignal method that chooses the correct frontend method to call
  
  TODO: add the possibility to bind the process to a certain processor
  """
  
  # incoming signals : from frontend to backend
  incoming_signal_defs={ # each key corresponds to a front- and backend methods
    "test_"    : {"test_int": int, "test_str": str},
    "stop_"    : []
    }
  
  # outgoing signals : from back to frontend.  Don't use same names as for incoming signals ..
  outgoing_signal_defs={
    "test_o"    : {"test_int": int, "test_str": str},
    }
  
  
  def __init__(self,name,**kwargs):
    super().__init__()
    self.pre=self.__class__.__name__+" : "+name+" : " # auxiliary string for debugging output
    self.name                 = name
    self.signal_in            = Event()
    self.signal_out           = Event()
    self.pipe, self.childpipe = Pipe() # communications pipe.  Frontend uses self.pipe, backend self.childpipe
  
    self.signal_in.clear()
    self.signal_out.clear()
    
    
  def getPipe(self):
    """Returns communication pipe for front-end
    """
    return self.pipe
    
    
  def preRun_(self):
    """After the fork, but before starting the process loop
    """
    pass
    
    
  def postRun_(self):
    """Just before process exit
    """
    print(self.pre,"post: bye!")
  
  
  def cycle_(self):
    # Do whatever your process should be doing, remember timeout every now and then
    time.sleep(5)
    print(self.pre,"hello!")
  
  
  def startAsThread(self):
    from threading import Thread
    t=Thread(target=self.run)
    t.start()
    
    
  def run(self): # No "_" in the name, but nevertheless, running in the backed
    """After the fork. Now the process starts running
    """
    self.preRun_()
    self.running=True
    
    while(self.running):
      self.cycle_()
      self.handleSignal_()
        
    self.postRun_()
    
    
  def handleSignal_(self):
    """Signals handling in the backend
    """
    if (self.signal_in.is_set()):
      signal_dic=self.childpipe.recv()
      method_name=signal_dic.pop("name")
      method=getattr(self,method_name)
      method(**signal_dic)
      self.signal_in.clear()
      self.signal_out.set()
    
    
  def sendSignal(self,**kwargs): # sendSignal(name="test",test_int=1,test_str="kokkelis")
    """Incoming signals: this is used by frontend methods to send signals to the backend
    """
    try:
      name=kwargs.pop("name")
    except KeyError:
      raise(AttributeError("Signal name missing"))
    
    model=self.incoming_signal_defs[name] # a dictionary: {"parameter_name" : parameter_type}
    
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
    
    
  def handleSignal(self,signal_dic):
    """Signal handling in the frontend
    """
    method_name=signal_dic.pop("name")
    method=getattr(self,method_name)
    method(**signal_dic)
    
    
  def sendSignal_(self,**kwargs): # sendSignal_(name="test_out",..)
    """Outgoing signals: signals from backend to frontend
    """
    try:
      name=kwargs.pop("name")
    except KeyError:
      raise(AttributeError("Signal name missing"))
    
    model=self.outgoing_signal_defs[name] # a dictionary: {"parameter_name" : parameter_type}
    
    for key in kwargs:
      model_type      =model[key] # raises error if user is using undefined signal
      parameter_type  =kwargs[key].__class__
      if (model_type==parameter_type):
        pass
      else:
        raise(AttributeError("Wrong type for parameter "+str(key)))
      
    kwargs["name"]=name
    
    self.childpipe.send(kwargs)
    
    
  # *** backend methods corresponding to each incoming signals ***
  
  def stop_(self):
    self.running=False
  
  
  def test_(self,test_int=0,test_str="nada"):
    print(self.pre,"test_ signal received with",test_int,test_str)
    
  
  # ** frontend methods corresponding to each incoming signal: these communicate with the backend via pipes **

  def stop(self):
    self.sendSignal(name="stop_")
  
  
  def test(self,**kwargs):
    dictionaryCheck(self.incoming_signal_defs["test_"],kwargs)
    kwargs["name"]="test_"
    self.sendSignal(**kwargs)
    
    
  # ** frontend methods corresponding to each outgoing signal **
  
  # typically, there is a QThread in the frontend-side reading the process pipe
  # the QThread reads kwargs dictionary from the pipe, say
  # {"name":"test_o", "test_str":"eka", "test_int":1}
  # And calls handleSignal(kwargs)
  
  def test_o(self,**kwargs):
    pass
    
    
    
    
    
class ValkkaShmemProcess(ValkkaProcess):
  """An example process, using the valkka shared memory client
  """
  
  incoming_signal_defs={ # each key corresponds to a front- and backend methods
    "test_"    : {"test_int": int, "test_str": str},
    "stop_"    : [],
    "ping_"    : {"message":str}
    }
  
  outgoing_signal_defs={
    "pong_o"    : {"message":str}
    }
  
  parameter_defs={
    "image_dimensions"       : (tuple,(1920//4,1080//4)),
    "n_ringbuffer"           : (int,10),                 # size of the ringbuffer
    "memname"                : str,
    "mstimeout"              : (int,1000),
    "verbose"                : (bool,False)
    }
  
  
  def __init__(self,name,**kwargs):
    super().__init__(name)
    parameterInitCheck(ValkkaShmemProcess.parameter_defs,kwargs,self) # check kwargs agains parameter_defs, attach ok'd parameters to this object as attributes
    typeCheck(self.image_dimensions[0],int)
    typeCheck(self.image_dimensions[1],int)
    self.n_bytes=self.image_dimensions[0]*self.image_dimensions[1]*3
    
    
  def preRun_(self):
    """After the fork, but before starting the process loop
    """
    self.client=ShmemClient(
      name        =self.memname,
      n_ringbuffer=self.n_ringbuffer,
      n_bytes     =self.n_bytes,
      mstimeout   =self.mstimeout,
      verbose     =self.verbose
      )

    
  def postRun_(self):
    """Just before process exit
    """
    print(self.pre,"post: bye!")
  
  
  def cycle_(self):
    index, isize = self.client.pull()
    if (index==None): # semaphore timed out
      print("Semaphore timeout")
    else:
      print("Current index, size=",index,isize)
      print("Payload=",self.client.shmem_list[index][0:min(isize,10)])
      
    
  # *** backend methods corresponding to incoming signals ***
  def stop_(self):
    self.running=False
  
  
  def test_(self,test_int=0,test_str="nada"):
    print(self.pre,"test_ signal received with",test_int,test_str)
    
  
  def ping_(self,message="nada"):
    print(self.pre,"At backend: ping_ received",message,"sending it back to front")
    self.sendSignal_(name="pong_o",message=message)
  
  
  # ** frontend methods launching incoming signals
  def stop(self):
    self.sendSignal(name="stop_")
  
  
  def test(self,**kwargs):
    dictionaryCheck(self.incoming_signal_defs["test_"],kwargs)
    kwargs["name"]="test_"
    self.sendSignal(**kwargs)
    
    
  def ping(self,**kwargs):
    dictionaryCheck(self.incoming_signal_defs["ping_"],kwargs)
    kwargs["name"]="ping_"
    self.sendSignal(**kwargs)
    
    
  # ** frontend methods handling received outgoing signals ***
  def pong_o(self,message="nada"):
    print("At frontend: pong got message",message)
  
  
  
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
  st="""Test ValkkaShmemProcess
  """
  pre=pre_mod+"test5 :"
  print(pre,st)
  
  p=ValkkaShmemProcess(
    "process1",
    image_dimensions = (1920//4,1080//4),
    n_ringbuffer     = 10,
    memname          = "testing",
    mstimeout        = 1000,
    verbose          = True
  )
  
  p.start()
  time.sleep(5)
  p.test(test_int=10,test_str="kikkelis")
  time.sleep(3)
  p.stop()
  


def test6():
  from threading import Thread
  
  st="""Test ValkkaShmemProcess with a thread running in the frontend
  """
  pre=pre_mod+"test6 :"
  print(pre,st)
  
  class FrontEndThread(Thread):
    
    def __init__(self,valkka_process):
      super().__init__()
      self.valkka_process=valkka_process
      
    def run(self):
      self.loop=True
      pipe=self.valkka_process.getPipe()
      while self.loop:
        st=pipe.recv() # get signal from the process
        self.valkka_process.handleSignal(st)
      
    def stop(self):
      self.loop=False
  
  
  p=ValkkaShmemProcess(
    "process1",
    image_dimensions = (1920//4,1080//4),
    n_ringbuffer     = 10,
    memname          = "testing",
    mstimeout        = 1000,
    verbose          = True
  )
  
  
  # thread running in front-end
  t=FrontEndThread(p)
  t.start()
  
  # process runs in its own virtual memory space - i.e. in the backend
  p.start()
  time.sleep(5)
  p.ping(message="<hello from front>")
  time.sleep(3)
  p.ping(message="<hello again from front>")
  time.sleep(3)
  t.stop()
  p.ping(message="<hello once again from front>")
  t.join()
  p.stop()
  

def test7():
  from threading import Thread
  import select
  
  st="""Several ValkkaShmemProcesses with a single frontend thread.
  """
  pre=pre_mod+"test7 :"
  print(pre,st)
  
  class FrontEndThread(Thread):
    
    def __init__(self,valkka_process,valkka_process2):
      super().__init__()
      self.valkka_process  =valkka_process
      self.valkka_process2 =valkka_process2
      self.process_by_pipe={
        self.valkka_process. getPipe() : self.valkka_process,
        self.valkka_process2.getPipe() : self.valkka_process2
        }
      
      
    def run(self):
      self.loop=True
      rlis=[]
      for key in self.process_by_pipe:
        rlis.append(key)
      wlis=[]
      elis=[]
      while self.loop:
        tlis=select.select(rlis,wlis,elis)
        for pipe in tlis[0]:
          p=self.process_by_pipe[pipe]
          # print("receiving from",p,"with pipe",pipe)
          st=pipe.recv() # get signal from the process
          # print("got from  from",p,"with pipe",pipe,":",st)
          p.handleSignal(st)
      
      
    def stop(self):
      self.loop=False
  
  
  p=ValkkaShmemProcess(
    "process1",
    image_dimensions = (1920//4,1080//4),
    n_ringbuffer     = 10,
    memname          = "testing",
    mstimeout        = 1000,
    verbose          = True
  )
  
  p2=ValkkaShmemProcess(
    "process2",
    image_dimensions = (1920//4,1080//4),
    n_ringbuffer     = 10,
    memname          = "testing2",
    mstimeout        = 1000,
    verbose          = True
  )
  
  # thread running in front-end
  t=FrontEndThread(p,p2)
  t.start()
  
  # process runs in its own virtual memory space - i.e. in the backend
  p. start()
  p2.start()
  time.sleep(5)
  
  p. ping(message="<hello from front>")
  p2.ping(message="<hello from front2>")
  time.sleep(3)
  
  p. ping(message="<hello again from front>")
  time.sleep(3)
  
  t.stop()
  
  p. ping(message="<hello once again from front>")
  p2.ping(message="<second hello from front2>")
  t.join()
  
  p.stop()
  p2.stop()

  
  
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



