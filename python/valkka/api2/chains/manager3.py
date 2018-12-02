"""
manager3.py : Managed filterchain classes, this time the right way.  Resources are managed hierarchically, decoding is turned off if its not required

 * Copyright 2018 Valkka Security Ltd. and Sampsa Riikonen.
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

@file    manage3.py
@author  Sampsa Riikonen
@date    2018
@version 0.10.0 

@brief   Managed filterchain classes, this time the right way.  Resources are managed hierarchically, decoding is turned off if its not required
"""

import sys
import time
import random
# so, everything that has .core, refers to the api1 level (i.e. swig
# wrapped cpp code)
from valkka import core
# api2 versions of the thread classes
from valkka.api2.threads import LiveThread, USBDeviceThread, OpenGLThread
from valkka.api2.tools import parameterInitCheck, typeCheck, generateGetters
from valkka.api2.chains.port import ViewPort

pre_mod = "valkka.api2.chains.manage3 : "


class ManagedFilterchain3:
    """This class implements the following filterchain:

    ::    
        main_branch:
                                                                                              +-->
                                                                                              |
        SOURCE                  -->> (AVThread:avthread) --> {ForkFrameFilterN:fork_filter} --+-->  .. OpenGLTreads, RenderContexts
                                                                                              |
                                                                                              +-->  swscale_branch (connect on demand)

        swscale_branch:
         
        {TimeIntervalFrameFilter:interval_filter} --> {SwScaleFrameFilter:sws_filter} --> {ForkFrameFilterN: sws_fork_filter}
                                                                                                         |
                                                                                            +------------+------------+             
                                                                                            |            |            |
                                                                                            
                                                                                            on-demand RGBShmemFrameFilter(s)
        
    OpenGLThread(s) and stream connections to windows (RenderContexts) are created upon request.
    Decoding at AVThread is turned on/off, depending if it is required downstream
    
    SOURCE is defined in the subclasses
    
    """ 

    parameter_defs = {
        "openglthreads": list,
        "address"      : str,  # string identifying the stream
        "slot"         : int,
        "_id"          : (int, 0),

        # these are for the AVThread instance:
        "n_basic": (int, 20),  # number of payload frames in the stack
        "n_setup": (int, 20),  # number of setup frames in the stack
        "n_signal": (int, 20),  # number of signal frames in the stack
        "flush_when_full": (bool, False),  # clear fifo at overflow

        "affinity": (int, -1),
        "verbose": (bool, False),
        "msreconnect": (int, 0),

        # Timestamp correction type: TimeCorrectionType_none,
        # TimeCorrectionType_dummy, or TimeCorrectionType_smart (default)
        "time_correction": None,
        
        "shmem_image_dimensions" : (tuple, (1920//4, 1080//4)),
        "shmem_n_buffer"         : (int, 10),
        "shmem_image_interval"   : (int, 1000)
    }

    def __init__(self, **kwargs):
        # auxiliary string for debugging output
        self.pre = self.__class__.__name__ + " : "
        # check for input parameters, attach them to this instance as
        # attributes
        parameterInitCheck(ManagedFilterchain3.parameter_defs, kwargs, self)
        generateGetters(self.parameter_defs, self)

        for openglthread in self.openglthreads:
            assert(issubclass(openglthread.__class__, OpenGLThread))
        self.init()


    def init(self):
        self.idst = str(id(self))

        # init the manager
        self.ports = []
        self.tokens_by_port = {}

        self.shmem_counter = 0
        self.shmem_terminals = {}
        self.width      = self.shmem_image_dimensions[0]
        self.height     = self.shmem_image_dimensions[1]

        self.makeChain()
        self.createContext()
        self.startThreads()
        self.active = True
        
        
    def report(self, *args):
        if (self.verbose):
            print(self.pre, *args)

    def getParDic(self, keys):
        """Get those parameters from parameters_defs that are in keys as well
        """
        dic={}
        for key in keys:
            if (key in self.parameter_defs):
                dic[key] = getattr(self, key)
        return dic
            
            
    def __del__(self):
        self.close()

    def __str__(self):
        return "<"+str(self.__class__.__name__)+": slot: %i | _id: %i | address: %s>" % (self.slot, self._id, self.address)

    def close(self):
        if (self.active):
            if (self.verbose):
                print(self.pre, "Closing threads and contexes")
            self.decodingOff()
            self.closeContext()
            self.stopThreads()
            self.active = False

    def requestClose(self):
        self.requestStopThreads()
        
    def waitClose(self):
        self.waitStopThreads()
        
    def makeChain(self):
        """Create the filter chain
        """
        # *** main_branch ***
        self.fork_filter = core.ForkFrameFilterN("av_fork_at_slot_" + str(
            self.slot))  # FrameFilter chains can be attached to ForkFrameFilterN after it's been instantiated

        self.framefifo_ctx = core.FrameFifoContext()
        self.framefifo_ctx.n_basic = self.n_basic
        self.framefifo_ctx.n_setup = self.n_setup
        self.framefifo_ctx.n_signal = self.n_signal
        self.framefifo_ctx.flush_when_full = self.flush_when_full

        self.avthread = core.AVThread(
            "avthread_" + self.idst,
            self.fork_filter,
            self.framefifo_ctx)
        if (self.verbose): print(self.pre,"binding AVThread to core", int(self.affinity))
        self.avthread.setAffinity(self.affinity)
        # get input FrameFilter from AVThread
        self.av_in_filter = self.avthread.getFrameFilter()
        
        # *** swscale_branch ***
        self.sws_fork_filter = core.ForkFrameFilterN("sws_fork_at_slot_" + str(self.slot))
        self.sws_filter      = core.SwScaleFrameFilter("sws_filter", self.width, self.height, self.sws_fork_filter)
        self.interval_filter = core.TimeIntervalFrameFilter("interval_filter", self.shmem_image_interval, self.sws_filter)
        
        # self.interval_filter = core.BriefInfoFrameFilter("interval_filter") # DEBUG
        # self.fork_filter.connect("swscale", self.interval_filter) # DEBUG: connect from the start
    
    
    def createContext(self):
        """Defined the connection and uses (feeds) self.av_in_filter with that connection
        """
        raise(AssertionError("virtual method"))
        
        
    def closeContext(self):
        """Close and deregister the connection
        """
        raise(AssertionError("virtual method"))
        
        
    def getShmem(self):
        """Returns the unique name identifying the shared mem and semaphores.  The name can be passed to the machine vision routines.
        """
        # if first time, connect main branch to swscale_branch
        if (len(self.shmem_terminals)<1):
            self.report("getShmem : connecting swscale_branch")
            self.fork_filter.connect("swscale", self.interval_filter)
        
        # shmem_name =self.idst + "_" + str(self.shmem_counter)
        shmem_name =self.idst + "_" + str(len(self.shmem_terminals))
        self.report( "getShmem : reserving", shmem_name)
        shmem_filter    =core.RGBShmemFrameFilter(shmem_name, self.shmem_n_buffer, self.width, self.height)
        # shmem_filter    =core.BriefInfoFrameFilter(shmem_name) # DEBUG: see if you are actually getting any frames here ..

        self.shmem_terminals[shmem_name] = shmem_filter
        self.sws_fork_filter.connect(shmem_name, shmem_filter)
    
        return shmem_name 
    
        
    def releaseShmem(self, shmem_name):
        try:
            self.shmem_terminals.pop(shmem_name)
        except KeyError:
            return
        self.report( "releaseShmem : releasing", shmem_name)
        self.sws_fork_filter.disconnect(shmem_name)
        
        # if no more shmem requests, disconnect swscale_branch
        if (len(self.shmem_terminals)==0): # that was the last one ..
            self.report( "getShmem : disconnecting swscale_branch")
            self.fork_filter.disconnect("swscale")
    
    def startThreads(self):
        """Starts thread required by the filter chain
        """
        self.avthread.startCall()

    def stopThreads(self):
        """Stops threads in the filter chain
        """
        self.avthread.stopCall()

    def requestStopThreads(self):
        self.avthread.requestStopCall()
                
    def waitStopThreads(self):
        self.avthread.waitStopCall()

    def decodingOff(self):
        self.avthread.decodingOffCall()

    def decodingOn(self):
        self.avthread.decodingOnCall()

    def addViewPort(self, view_port):
        assert(issubclass(view_port.__class__, ViewPort))
        # ViewPort object is created by the widget .. and stays alive while the
        # widget exists.

        window_id = view_port.getWindowId()
        x_screen_num = view_port.getXScreenNum()
        openglthread = self.openglthreads[x_screen_num]

        if (self.verbose):
            print(self.pre,
                "addViewPort: view_port, window_id, x_screen_num",
                view_port,
                window_id,
                x_screen_num)

        if (view_port in self.ports):
            # TODO: implement == etc. operators : compare window_id (and
            # x_screen_num) .. nopes, if the object stays the same
            self.delViewPort(view_port)

        # run through all ViewPort instances in self.ports to find the number
        # of x-screen requests
        n_x_screen_ports = self.getNumXscreenPorts(x_screen_num)

        if (n_x_screen_ports < 1):
            # this only in the first time : start sending frames to X screen
            # number x_screen_num!
            if (self.verbose):
                print(self.pre,
                    "addViewPort: start streaming to x-screen",
                    x_screen_num)
            self.fork_filter.connect(
                "openglthread_" + str(x_screen_num),
                openglthread.getInput())

        # send frames from this slot to correct openglthread and window_id
        print(self.pre, "       connecting slot, window_id", self.slot, window_id)
        token = openglthread.connect(slot=self.slot, window_id=window_id)
        print(self.pre, "       ==> connected slot, window_id, token", self.slot, window_id, token)
        self.tokens_by_port[view_port] = token

        if (len(self.ports) < 1):
            # first request for this stream : time to start decoding!
            if (self.verbose):
                print(self.pre, "addViewPort: start decoding slot", self.slot)
            self.avthread.decodingOnCall()

        self.ports.append(view_port)

    def delViewPort(self, view_port):
        assert(issubclass(view_port.__class__, ViewPort))

        window_id = view_port.getWindowId()
        x_screen_num = view_port.getXScreenNum()
        openglthread = self.openglthreads[x_screen_num]

        if (self.verbose):
            print(self.pre,
                "delViewPort: view_port, window_id, x_screen_num",
                view_port,
                window_id,
                x_screen_num)

        if (view_port not in self.ports):
            print(self.pre, "delViewPort : FATAL : no such port", view_port)
            return

        self.ports.remove(view_port)  # remove this port from the list
        # remove the token associated to x-window output
        token = self.tokens_by_port.pop(view_port)
        # stop the slot => render context / x-window mapping associated to the
        # token
        print(self.pre, "delViewPort:       disconnecting token", token)
        openglthread.disconnect(token)
        print(self.pre, "delViewPort:       OK disconnected token", token)

        n_x_screen_ports = self.getNumXscreenPorts(x_screen_num)

        if (n_x_screen_ports < 1):
            # no need to send this stream to X Screen number x_screen_num
            if (self.verbose):
                print(self.pre,
                    "delViewPort: removing stream from x-screen",
                    x_screen_num)

            self.fork_filter.disconnect("openglthread_" + str(x_screen_num))

        if (len(self.ports) < 1):
            # no need to decode the stream anymore
            self.avthread.decodingOffCall()

    def getNumXscreenPorts(self, x_screen_num):
        """Run through ViewPort's, count how many of them are using X screen number x_screen_num
        """
        sm = 0
        for view_port in self.ports:
            if (issubclass(view_port.__class__, ViewPort)):
                if (view_port.getXScreenNum() == x_screen_num):
                    sm += 1
        if (self.verbose):
            print(
                self.pre,
                "getNumXscreenPorts: slot",
                self.slot,
                "serves",
                sm + 1,
                "view ports")
        return sm


    def setBoundingBoxes(self, view_port, bbox_list):
        x_screen_num = view_port.getXScreenNum()
        openglthread = self.openglthreads[x_screen_num]

        if (view_port in self.tokens_by_port):
            token = self.tokens_by_port[view_port]
            openglthread.core.clearObjectsCall(token)
            for bbox in bbox_list:
                openglthread.core.addRectangleCall(token, bbox[0], bbox[1], bbox[2], bbox[3]) # left, right, top, bottom


            
class LiveManagedFilterchain(ManagedFilterchain3):
    
    
    parameter_defs = {
        "livethread"        : LiveThread,
        # Operating system socket ringbuffer size in bytes # 0 means default
        "recv_buffer_size"  : (int, 0),
        # Reordering buffer time for Live555 packets in MILLIseconds # 0 means default
        "reordering_mstime" : (int, 0)
    }
    parameter_defs.update(ManagedFilterchain3.parameter_defs)
    
    
    def __init__(self, **kwargs):
        parameterInitCheck(LiveManagedFilterchain.parameter_defs, kwargs, self) # checks that parameters ok, attaches them as attributes to self
        # remove parameters not used by the mother class
        kwargs.pop("livethread")
        if ("recv_buffer_size"  in kwargs): kwargs.pop("recv_buffer_size") 
        if ("reordering_mstime" in kwargs): kwargs.pop("reordering_mstime")
        super().__init__(**kwargs)
    
    def createContext(self):
        """Creates a LiveConnectionContext and registers it to LiveThread
        """
        # define stream source, how the stream is passed on, etc.

        self.ctx = core.LiveConnectionContext()
        # slot number identifies the stream source
        self.ctx.slot = self.slot

        if (self.address.find("rtsp://") == 0):
            self.ctx.connection_type = core.LiveConnectionType_rtsp
        else:
            self.ctx.connection_type = core.LiveConnectionType_sdp  # this is an rtsp connection

        self.ctx.address = self.address
        # stream address, i.e. "rtsp://.."

        self.ctx.framefilter = self.av_in_filter

        self.ctx.msreconnect = self.msreconnect

        # some extra parameters
        """
        // ctx.time_correction =TimeCorrectionType::none;
        // ctx.time_correction =TimeCorrectionType::dummy;
        // default time correction is smart
        // ctx.recv_buffer_size=1024*1024*2;  // Operating system ringbuffer size for incoming socket
        // ctx.reordering_time =100000;       // Live555 packet reordering treshold time (microsecs)
        """
        if (self.time_correction is not None):
            self.ctx.time_correction = self.time_correction
        self.ctx.recv_buffer_size = self.recv_buffer_size
        self.ctx.reordering_time = self.reordering_mstime * 1000  # from millisecs to microsecs

        # send the information about the stream to LiveThread
        self.livethread.registerStream(self.ctx)
        self.livethread.playStream(self.ctx)


    def closeContext(self):
        """Close and deregister the connection
        """
        self.livethread.stopStream(self.ctx)
        self.livethread.deregisterStream(self.ctx)
    


class USBManagedFilterchain(ManagedFilterchain3):
    
    parameter_defs = {
        "usbthread" : USBDeviceThread
    }
    parameter_defs.update(ManagedFilterchain3.parameter_defs)
    
    
    def __init__(self, **kwargs):
        parameterInitCheck(USBManagedFilterchain.parameter_defs, kwargs, self) # checks that parameters ok, attaches them as attributes to self
        # remove parameters not used by the mother class
        kwargs.pop("usbthread")
        super().__init__(**kwargs)
    
    def createContext(self):
        """Creates a LiveConnectionContext and registers it to LiveThread
        """
        # define stream source, how the stream is passed on, etc.
        
        # USBCameraConnectionContext
        """
        std::string device;                                           // <pyapi>
        /** A unique stream slot that identifies this stream */
        SlotNumber         slot;                                      // <pyapi>
        /** Frames are feeded into this FrameFilter */
        FrameFilter*       framefilter;                               // <pyapi>
        int                width;                                     // <pyapi>
        int                height;                                    // <pyapi>
        /** How to perform frame timestamp correction */
        TimeCorrectionType time_correction;                           // <pyapi>
        // TODO: format, fps, etc.                                    // <pyapi>
        """
        
        
        self.ctx = core.USBCameraConnectionContext()
        self.ctx.device      = self.address
        self.ctx.width       = 1280
        self.ctx.height      = 720
        self.ctx.slot        = self.slot
        self.ctx.framefilter = self.av_in_filter
        if (self.time_correction is not None):
            self.ctx.time_correction = self.time_correction
        
        self.usbthread.playStream(self.ctx)


    def closeContext(self):
        """Close and deregister the connection
        """
        self.usbthread.stopStream(self.ctx)


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

