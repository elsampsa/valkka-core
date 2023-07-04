"""
valkkafs.py : Framefilter chains for simultaneous decoding, presenting and reading / writing frames to ValkkaFS

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

@file    valkkafs.py
@author  Sampsa Riikonen
@date    2017
@version 1.4.0 

@brief Framefilter chains for simultaneous decoding, presenting and reading / writing frames to ValkkaFS
"""

import sys
import time
import random
# so, everything that has .core, refers to the api1 level (i.e. swig
# wrapped cpp code)
from valkka import core
# api2 versions of the thread classes
from valkka.api2.threads import LiveThread
from valkka.api2.tools import parameterInitCheck, typeCheck
from valkka.api2.valkkafs import ValkkaFSManager
pre_mod = __name__


class ValkkaFSLiveFilterchain:
    """This class implements the following filterchain:
    
    ValkkaFSLiveFilterchain:

    ::
                                                                  +---> (AVThread:avthread) -------> {ForkFrameFilterN:fork_yuv} -------+
                                                                  |                                                                     |
      (LiveThread:livethread) --> {ForkFrameFilterN:fork} --------+ request forked H264 stream                                          +--->> OpenGLThread (connected by user)
                                                                  |                                                                     |
                                       ValkkaFSWriterThread <<----+                                                                     + request forked YUV

    ValkkaFSFileFilterchain:
    
    
    ::
    
      ValkkaFSReaderThread -->> FileCacherThread ------------->> {ForkFrameFilterN:fork} ------+
                                                                                               |
                                  {ForkFrameFilterN:fork_yuv} <-- (AVThread:avthread) <<-------+
                                            |                                                  |
                                            |                                                  + request forked H264 stream
                                            +--->>> OpenGLThread (connected by user)
                                            |
                                            + request forked YUV


      ValkkaFSManager:

      - ValkkaFSWriterThread
      - ValkkaFSReaderThread
      - FileCacherThread

    """

    parameter_defs = {
        "livethread": LiveThread,
        "valkkafsmanager": ValkkaFSManager,
        "address": str,
        "slot": int,
        "id_rec": int,

        # these are for the AVThread instance:
        "n_basic": (int, 20),  # number of payload frames in the stack
        "n_setup": (int, 20),  # number of setup frames in the stack
        "n_signal": (int, 20),  # number of signal frames in the stack
        
        "affinity": (int, -1),
        "verbose": (bool, False),
        "msreconnect": (int, 0),

        # Timestamp correction type: TimeCorrectionType_none,
        # TimeCorrectionType_dummy, or TimeCorrectionType_smart (default)
        "time_correction": None,
        # Operating system socket ringbuffer size in bytes # 0 means default
        "recv_buffer_size": (int, 0),
        # Reordering buffer time for Live555 packets in MILLIseconds # 0 means
        # default
        "reordering_mstime": (int, 0)
    }

    def __init__(self, **kwargs):
        # auxiliary string for debugging output
        self.pre = self.__class__.__name__ + " : "
        # check for input parameters, attach them to this instance as
        # attributes
        parameterInitCheck(self.parameter_defs, kwargs, self)
        self.init()

    def init(self):
        self.idst = str(id(self))
        self.makeChain()
        self.startThreads()
        self.active = True

    def __del__(self):
        self.close()

    def close(self):
        if (self.active):
            if (self.verbose):
                print(self.pre, "Closing threads and contexes")
            self.decodingOff()
            self.closeLiveContext()
            self.stopThreads()
            self.active = False

    def makeChain(self):
        """Create the filterchains
        """
        self.fork = core.ForkFrameFilterN("fork_" + str(self.slot))
        self.fork_yuv = core.ForkFrameFilterN("fork_yuv_" + str(self.slot))

        self.framefifo_ctx = core.FrameFifoContext()
        self.framefifo_ctx.n_basic = self.n_basic
        self.framefifo_ctx.n_setup = self.n_setup
        self.framefifo_ctx.n_signal = self.n_signal
        self.framefifo_ctx.flush_when_full = True

        self.avthread = core.AVThread(
            "avthread_" + self.idst,
            self.fork_yuv,                          # writes to self.fork_yuv
            self.framefifo_ctx)
        
        self.avthread.setAffinity(self.affinity)

        # initial connections : live stream
        self.createLiveContext() # LiveThread writes to self.fork
        self.connect_to_stream("live_decode_"+str(self.slot), self.avthread.getFrameFilter()) # self.fork to AVThread
        self.connect_to_stream("recorder_"+str(self.slot), self.valkkafsmanager.getFrameFilter()) # self.fork to ValkkaFSWriterThread
        self.valkkafsmanager.setInput(self.id_rec, self.slot)
        
        
    def connect_to_stream(self, name, framefilter):
        return self.fork.connect(name, framefilter)

    def connect_to_yuv(self, name, framefilter):
        return self.fork_yuv.connect(name, framefilter)

    def disconnect_from_stream(self, name):
        return self.fork.disconnect(name)
    
    def disconnect_from_yuv(self, name):
        return self.fork_yuv.disconnect(name)
    
    
    def createLiveContext(self):
        """Creates a LiveConnectionContext and registers it to LiveThread
        """
        self.ctx = core.LiveConnectionContext()
        self.ctx.slot = self.slot

        if (self.address.find("rtsp://") == 0):
            self.ctx.connection_type = core.LiveConnectionType_rtsp
        else:
            self.ctx.connection_type = core.LiveConnectionType_sdp

        self.ctx.address = self.address
        self.ctx.framefilter = self.fork       # writes to self.fork
        self.ctx.msreconnect = self.msreconnect

        # some extra parameters
        """
        ctx.time_correction =TimeCorrectionType::none;
        ctx.time_correction =TimeCorrectionType::dummy;
        default time correction is smart
        ctx.recv_buffer_size=1024*1024*2;  // Operating system ringbuffer size for incoming socket
        ctx.reordering_time =100000;       // Live555 packet reordering treshold time (microsecs)
        """
        if (self.time_correction is not None):
            self.ctx.time_correction = self.time_correction
        self.ctx.recv_buffer_size = self.recv_buffer_size
        self.ctx.reordering_time = self.reordering_mstime * 1000  # from millisecs to microsecs

        # send the information about the stream to LiveThread
        self.livethread.registerStream(self.ctx)
        self.livethread.playStream(self.ctx)


    def closeLiveContext(self):
        self.livethread.stopStream(self.ctx)
        self.livethread.deregisterStream(self.ctx)


    def startThreads(self):
        """Starts thread required by the filter chain
        """
        self.avthread.startCall()
        
    def stopThreads(self):
        """Stops threads in the filter chain
        """
        self.avthread.stopCall()
        
    def decodingOff(self):
        self.avthread.decodingOffCall()

    def decodingOn(self):
        self.avthread.decodingOnCall()



class ValkkaFSFileFilterchain:
    """This class implements the following filterchain:
    
    ValkkaFSLiveFilterchain:

    ::
                                                                  +---> (AVThread:avthread) -------> {ForkFrameFilterN:fork_yuv} -------+
                                                                  |                                                                     |
      (LiveThread:livethread) --> {ForkFrameFilterN:fork} --------+ request forked H264 stream                                          +--->> OpenGLThread (connected by user)
                                                                  |                                                                     |
                                       ValkkaFSWriterThread <<----+                                                                     + request forked YUV

    ValkkaFSFileFilterchain:
    
    
    ::
    
      ValkkaFSReaderThread -->> FileCacherThread ------------->> {ForkFrameFilterN:fork} ------+
                                                                                               |
                                  {ForkFrameFilterN:fork_yuv} <-- (AVThread:avthread) <<-------+
                                            |                                                  |
                                            |                                                  + request forked H264 stream
                                            +--->>> OpenGLThread (connected by user)
                                            |
                                            + request forked YUV


      ValkkaFSManager:

      - ValkkaFSWriterThread
      - ValkkaFSReaderThread
      - FileCacherThread

    """

    parameter_defs = {
        "valkkafsmanager": ValkkaFSManager,
        "slot": int,
        "id_rec": int,

        # these are for the AVThread instance:
        "n_basic": (int, 20),  # number of payload frames in the stack
        "n_setup": (int, 20),  # number of setup frames in the stack
        "n_signal": (int, 20),  # number of signal frames in the stack
        
        "affinity": (int, -1),
        "verbose": (bool, False),
        "msreconnect": (int, 0)
    }

    def __init__(self, **kwargs):
        # auxiliary string for debugging output
        self.pre = self.__class__.__name__ + " : "
        # check for input parameters, attach them to this instance as
        # attributes
        parameterInitCheck(self.parameter_defs, kwargs, self)
        self.init()

    def init(self):
        self.idst = str(id(self))
        self.makeChain()
        self.startThreads()
        self.active = True

    def __del__(self):
        self.close()

    def close(self):
        if (self.active):
            if (self.verbose):
                print(self.pre, "Closing threads and contexes")
            self.decodingOff()
            self.valkkafsmanager.clearOutput(self.ctx)
            self.stopThreads()
            self.active = False

    def makeChain(self):
        """Create the filterchains
        """
        self.fork = core.ForkFrameFilterN("fork_" + str(self.slot))
        self.fork_yuv = core.ForkFrameFilterN("fork_yuv_" + str(self.slot))

        self.framefifo_ctx = core.FrameFifoContext()
        self.framefifo_ctx.n_basic = self.n_basic
        self.framefifo_ctx.n_setup = self.n_setup
        self.framefifo_ctx.n_signal = self.n_signal
        self.framefifo_ctx.flush_when_full = True

        self.avthread = core.AVThread(
            "avthread_" + self.idst,
            self.fork_yuv,                          # writes to self.fork_yuv
            self.framefifo_ctx)
        
        self.avthread.setAffinity(self.affinity)

        self.info = core.InfoFrameFilter("debug")

        # initial connections : recorded stream
        
        self.connect_to_stream("rec_decode_"+str(self.slot), self.avthread.getBlockingFrameFilter()) # self.fork to AVThread
        # self.connect_to_stream("rec_decode_"+str(self.slot), self.info) # debug 
        
        # # self.valkkafs.setOutput(_id, slot, framefilter)
        self.ctx = self.valkkafsmanager.setOutput(self.id_rec, self.slot, self.fork) # recorded stream to self.fork
        
        # self.connect_to_yuv("debug", self.info) # debug
        
        
    def connect_to_stream(self, name, framefilter):
        return self.fork.connect(name, framefilter)

    def connect_to_yuv(self, name, framefilter):
        return self.fork_yuv.connect(name, framefilter)

    def disconnect_from_stream(self, name):
        return self.fork.disconnect(name)
    
    def disconnect_from_yuv(self, name):
        return self.fork_yuv.disconnect(name)
    
    
    def startThreads(self):
        """Starts thread required by the filter chain
        """
        self.avthread.startCall()
        
    def stopThreads(self):
        """Stops threads in the filter chain
        """
        self.avthread.stopCall()
        
    def decodingOff(self):
        self.avthread.decodingOffCall()

    def decodingOn(self):
        self.avthread.decodingOnCall()



def main():
    pre = pre_mod + "main :"
    print(pre, "main: arguments: ", sys.argv)
    if (len(sys.argv) < 2):
        print(pre, "main: needs test number")
    else:
        st = "test" + str(sys.argv[1]) + "()"
        exec(st)


if (__name__ == "__main__"):
    main()
