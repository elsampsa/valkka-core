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
@version 0.17.0 

@brief   Encapsulation for Valkka's cpp shared memory client
"""
import logging
from valkka import core
from valkka.api2.tools import *

pre_mod = "valkka.api2.shmem: "


class ShmemClient:
    """A shared memory ringbuffer client.  The idea is here, that the ringbuffer "server" is instantiated at the cpp side.  Client must have exactly the same name (that identifies the shmem segments), and the number of ringbuffer elements.

    :param name:               name identifying the shared mem segment.  Must be same as in the server side.
    :param n_ringbuffer:       Number of elements in the ringbuffer.  Must be same as in the server side.
    :param n_bytes:            Size of each element in the ringbuffer.
    :param mstimeout:          Timeout for semaphores.  Default=0=no timeout.
    :param verbose:            Be verbose or not.  Default=False.

    """

    parameter_defs = {
        # :param name:               name identifying the shared mem segment.  Must be same as in the server side.
        "name": str,
        # :param n_ringbuffer:       Number of elements in the ringbuffer.  Must be same as in the server side.
        "n_ringbuffer": (int, 10),
        # :param n_bytes:            Size of each element in the ringbuffer.
        "n_bytes": int,
        # :param mstimeout:          Timeout for semaphores.  Default=0=no timeout.
        "mstimeout": (int, 0),
        # :param verbose:            Be verbose or not.  Default=False.
        "verbose": (bool, False)
    }

    def __init__(self, **kwargs):
        # auxiliary string for debugging output
        self.pre = self.__class__.__name__
        self.logger = getLogger(self.pre)
        # check kwargs agains parameter_defs, attach ok'd parameters to this
        # object as attributes
        parameterInitCheck(ShmemClient.parameter_defs, kwargs, self)

        self.index_p = core.new_intp()
        self.isize_p = core.new_intp()

        # print(self.pre,"shmem name=",self.name)
        # shmem ring buffer on the client side
        self.core = core.SharedMemRingBuffer(
            self.name, self.n_ringbuffer, self.n_bytes, self.mstimeout, False)

        """ # legacy
        self.shmem_list = []
        for i in range(self.n_ringbuffer):
            # if you're looking for this, it's defined in the .i swig interface file.
            # :)
            self.shmem_list.append(core.getNumpyShmem(self.core, i))
        """
        self.shmem_list = self.core.getBufferListPy()


    def setDebug(self):
        setLogger(self.logger, logging.DEBUG)


    def pull(self):
        """If semaphore was timed out (i.e. nothing was written to the ringbuffer) in mstimeout milliseconds, returns: None, None.  Otherwise returns the index of the shmem segment and the size of data written.
        """
        got = self.core.clientPull(self.index_p, self.isize_p)
        index = core.intp_value(
            self.index_p)
        isize = core.intp_value(self.isize_p)
        if (self.verbose):
            self.logger.debug("current index, size= %s %s", index, isize)
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

    parameter_defs = {
        # :param name:               name identifying the shared mem segment.  Must be same as in the server side.
        "name": str,
        # :param n_ringbuffer:       Number of elements in the ringbuffer.  Must be same as in the server side.
        "n_ringbuffer": (int, 10),
        "width": int,         # :param width:              RGB image width
        "height": int,         # :param height:             RGb image height
        # :param mstimeout:          Timeout for semaphores.  Default=0=no timeout.
        "mstimeout": (int, 0),
        # :param verbose:            Be verbose or not.  Default=False.
        "verbose": (bool, False)
    }

    def __init__(self, **kwargs):
        # auxiliary string for debugging output
        self.pre = self.__class__.__name__
        self.logger = getLogger(self.pre)
        # check kwargs agains parameter_defs, attach ok'd parameters to this
        # object as attributes
        parameterInitCheck(ShmemRGBClient.parameter_defs, kwargs, self)

        self.index_p = core.new_intp()
        self.isize_p = core.new_intp()
        # self.rgb_meta = RGB24Meta()

        # print(self.pre,"shmem name=",self.name)
        # shmem ring buffer on the client side
        self.core = core.SharedMemRingBufferRGB(
            self.name,
            self.n_ringbuffer,
            self.width,
            self.height,
            self.mstimeout,
            False)

        """ # legacy
        self.shmem_list = []
        for i in range(self.n_ringbuffer):
            # if you're looking for this, it's defined in the .i swig interface file.
            # :)
            self.shmem_list.append(core.getNumpyShmem(self.core, i))
        """
        
        #"""
        #print(self.pre,"shmem get list")
        self.shmem_list = self.core.getBufferListPy()
        #print(self.pre,"shmem got list")
        #"""


    def setDebug(self):
        setLogger(self.logger, logging.DEBUG)


    def useEventFd(self, event_fd):
        self.core.clientUseFd(event_fd)


    def pull(self):
        """If semaphore was timed out (i.e. nothing was written to the ringbuffer) in mstimeout milliseconds, returns: None, None.  Otherwise returns the index of the shmem segment and the size of data written.
        """
        got = self.core.clientPull(self.index_p, self.isize_p)
        index = core.intp_value(
            self.index_p)
        isize = core.intp_value(self.isize_p)
        # if (self.verbose):
        self.logger.debug("current index, size= %s %s", index, isize)
        if (got):
            return index, isize
        else:
            return None, None

    def pullFrame(self):
        """If semaphore was timed out (i.e. nothing was written to the ringbuffer) in mstimeout milliseconds, returns: None, None.  Otherwise returns the index of the shmem segment and the size of data written.
        """
        self.rgb_meta = core.RGB24Meta()
        got = self.core.clientPullFrame(self.index_p, self.rgb_meta)
        index = core.intp_value(self.index_p)
        # if (self.verbose):
        self.logger.debug("current index %s", index) # , ShmemRGBClient.metaToString(self.rgb_meta))
        # print(">", ShmemRGBClient.metaToString(self.rgb_meta))
        if (got):
            return index, self.rgb_meta
        else:
            return None, None

    def pullFrameThread(self):
        """Use with multithreading
        """
        self.rgb_meta = core.RGB24Meta()
        got = self.core.clientPullFrameThread(self.index_p, self.rgb_meta)
        index = core.intp_value(self.index_p)
        # if (self.verbose):
        self.logger.debug("current index %s ", index) # ShmemRGBClient.metaToString(self.rgb_meta))
        if (got):
            return index, self.rgb_meta
        else:
            return None, None

    @staticmethod
    def metaToString(rgb_meta):
        """
        std::size_t size; ///< Actual size copied       // <pyapi>
        int width;                                      // <pyapi>
        int height;                                     // <pyapi>
        SlotNumber slot;                                // <pyapi>
        long int mstimestamp;                           // <pyapi>
        """
        return "width = %i / height = %i / slot = %i / mstimestamp = %i / size = %i" % \
            (rgb_meta.width, rgb_meta.height, rgb_meta.slot, rgb_meta.mstimestamp, rgb_meta.size)



class ShmemRGBServer:
    """A shared memory ringbuffer server for RGB images.  

    :param name:               name identifying the shared mem segment.  Must be same as in the server side.
    :param n_ringbuffer:       Number of elements in the ringbuffer.  Must be same as in the server side.
    :param width:              RGB image width
    :param height:             RGb image height
    :param verbose:            Be verbose or not.  Default=False.

    """

    parameter_defs = {
        # :param name:               name identifying the shared mem segment.  Must be same as in the server side.
        "name"        : str,
        # :param n_ringbuffer:       Number of elements in the ringbuffer.  Must be same as in the server side.
        "n_ringbuffer": (int, 10),
        "width"       : int,         # :param width:              RGB image width
        "height"      : int,         # :param height:             RGb image height
        # :param mstimeout:          Timeout for semaphores.  Default=0=no timeout.
        "verbose"     : (bool, False),
        "use_event_fd": (bool, False)
    }

    def __init__(self, **kwargs):
        # auxiliary string for debugging output
        self.pre = self.__class__.__name__
        self.logger = getLogger(self.pre)
        # check kwargs agains parameter_defs, attach ok'd parameters to this
        # object as attributes
        parameterInitCheck(ShmemRGBServer.parameter_defs, kwargs, self)
        # shmem ring buffer on the server side
        self.core = core.SharedMemRingBufferRGB(
            self.name,
            self.n_ringbuffer,
            self.width,
            self.height,
            1000, # dummy value
            True) # True indicates server-side
        
        if self.use_event_fd:
            self.core.useEventFd()


    def setDebug(self):
        setLogger(self.logger, logging.DEBUG)


    def useEventFd(self, event_fd):
        self.core.serverUseFd(event_fd)


    def pushFrame(self, frame, slot, mstimestamp):
        # if self.verbose: print("pushFrame: ", frame.shape)
        ok = self.core.serverPushPyRGB(
            frame,
            slot, 
            mstimestamp)





def test4():
    st = """Test shmem client.  First start the cpp test program (SharedMemRingBuffer server side) with './shmem_test 3 0'
  """
    # at cpp side:
    # SharedMemRingBuffer rb("testing",10,30*1024*1024,1000,true); // name,
    # ncells, bytes per cell, timeout, server or not
    pre = pre_mod + "test4 :"
    print(pre, st)

    client = ShmemClient(
        name="testing",
        n_ringbuffer=10,
        n_bytes=30 * 1024 * 1024,
        mstimeout=0,
        verbose=True
    )

    while(True):
        i = int(input("number of buffers to read. 0 exits>"))
        if (i < 1):
            break
        while(i > 0):
            index, isize = client.pull()
            print("Current index, size=", index, isize)
            print("Payload=", client.shmem_list[index][0:min(isize, 10)])
            i -= 1


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
