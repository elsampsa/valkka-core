"""
threads.py : api level 1 => api level 2 encapsulation for LiveThread and OpenGLThread

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

@file    threads.py
@author  Sampsa Riikonen
@date    2017
@version 1.0.2 

@brief api level 1 => api level 2 encapsulation for LiveThread and OpenGLThread
"""
import time
from valkka import core
from valkka.api2.tools import *

core.ValkkaXInitThreads()

pre_mod = "valkka.api2.threads: "


class Namespace:  # a generic namespace ..

    def __init__(self):
        pass


class LiveThread:

    parameter_defs = {
        "name": (str, "live_thread"),
        "n_basic": (int, 20),  # number of payload frames in the stack
        "n_setup": (int, 20),  # number of setup frames in the stack
        "n_signal": (int, 20),  # number of signal frames in the stack
        "flush_when_full": (bool, False),  # clear fifo at overflow
        "affinity": (int, -1),
        "verbose": (bool, False),
        "rtsp_server": (int, -1)  # rtsp server portnum
    }

    def __init__(self, **kwargs):
        # auxiliary string for debugging output
        self.pre = self.__class__.__name__ + " : "
        # checks kwargs agains parameter_defs, attach ok'd parameters to this
        # object as attributes
        parameterInitCheck(LiveThread.parameter_defs, kwargs, self)

        # some "api-level 1" objects here = swig-wrapped cpp objects
        #
        # This parameter set is defined at the cpp header file "framefifo.h"
        self.framefifo_ctx = core.FrameFifoContext()
        self.framefifo_ctx.n_basic = self.n_basic
        self.framefifo_ctx.n_setup = self.n_setup
        self.framefifo_ctx.n_signal = self.n_signal
        self.framefifo_ctx.flush_when_full = self.flush_when_full
        # swig wrapped cpp LiveThread
        self.core = core.LiveThread(self.name, self.framefifo_ctx)
        self.core.setAffinity(self.affinity)
        if (self.rtsp_server > -1):
            self.core.setRTSPServer(self.rtsp_server)

        self.input_filter = self.core.getFrameFilter()

        self.active = True
        self.core.startCall()

    def getInput(self):
        return self.input_filter

    def registerStream(self, ctx):
        self.core.registerStreamCall(ctx)

    def deregisterStream(self, ctx):
        self.core.deregisterStreamCall(ctx)

    def playStream(self, ctx):
        self.core.playStreamCall(ctx)

    def stopStream(self, ctx):
        self.core.stopStreamCall(ctx)

    def close(self):
        if not self.active:
            return
        if (self.verbose):
            print(self.pre, "stopping core.LiveThread")
        self.core.stopCall()
        self.active = False

    def requestClose(self):
        self.core.requestStopCall()
        
    def waitClose(self):
        self.core.waitStopCall()
        self.active = False


    # """
    def __del__(self):
        self.close()
    # """



class USBDeviceThread:
    
    parameter_defs = {
        "name":     (str, "usb_thread"),
        "affinity": (int, -1),
        "verbose":  (bool, False)
    }

    def __init__(self, **kwargs):
        # auxiliary string for debugging output
        self.pre = self.__class__.__name__ + " : "
        # checks kwargs agains parameter_defs, attach ok'd parameters to this
        # object as attributes
        parameterInitCheck(LiveThread.parameter_defs, kwargs, self)

        # swig wrapped cpp USBDeviceThread
        self.core = core.USBDeviceThread(self.name)
        self.core.setAffinity(self.affinity)
        self.active = True
        self.core.startCall()

    """
    def registerStream(self, ctx):
        pass

    def deregisterStream(self, ctx):
        pass
    """

    def playStream(self, ctx):
        self.core.playCameraStreamCall(ctx)

    def stopStream(self, ctx):
        self.core.stopCameraStreamCall(ctx)

    def close(self):
        if not self.active:
            return
        if (self.verbose):
            print(self.pre, "stopping core.USBDeviceThread")
        self.core.stopCall()
        self.active = False

    def requestClose(self):
        self.core.requestStopCall()
        
    def waitClose(self):
        self.core.waitStopCall()
        self.active = False


class FileThread:
    # FileThread(const char* name, int core_id=-1);

    parameter_defs = {
        "name": (str, "file_thread"),
        "n_basic": (int, 20),  # number of payload frames in the stack
        "n_setup": (int, 20),  # number of setup frames in the stack
        "n_signal": (int, 20),  # number of signal frames in the stack
        "flush_when_full": (bool, False),  # clear fifo at overflow
        "affinity": (int, -1),
        "verbose": (bool, False)
    }

    def __init__(self, **kwargs):
        # auxiliary string for debugging output
        self.pre = self.__class__.__name__ + " : "
        # checks kwargs agains parameter_defs, attach ok'd parameters to this
        # object as attributes
        parameterInitCheck(FileThread.parameter_defs, kwargs, self)

        # some "api-level 1" objects here = swig-wrapped cpp objects
        #
        # This parameter set is defined at the cpp header file "framefifo.h"
        self.framefifo_ctx = core.FrameFifoContext()
        self.framefifo_ctx.n_basic = self.n_basic
        self.framefifo_ctx.n_setup = self.n_setup
        self.framefifo_ctx.n_signal = self.n_signal
        self.framefifo_ctx.flush_when_full = self.flush_when_full
        # swig wrapped cpp FileThread
        self.core = core.FileThread(self.name, self.framefifo_ctx)
        self.core.setAffinity(self.affinity)

        self.input_filter = self.core.getFrameFilter()

        self.active = True
        self.core.startCall()

    """
  void closeFileStreamCall      (FileContext &file_ctx); ///< API method: registers a stream                                // <pyapi>
  void openFileStreamCall       (FileContext &file_ctx); ///< API method: de-registers a stream                             // <pyapi>
  void seekFileStreamCall       (FileContext &file_ctx); ///< API method: seek to a certain point                           // <pyapi>
  void playFileStreamCall       (FileContext &file_ctx); ///< API method: starts playing the stream and feeding frames      // <pyapi>
  void stopFileStreamCall       (FileContext &file_ctx); ///< API method: stops playing the stream and feeding frames       // <pyapi>
  """

    def getInput(self):
        return self.input_filter

    def openStream(self, ctx):
        self.core.openFileStreamCall(ctx)

    def closeStream(self, ctx):
        self.core.closeFileStreamCall(ctx)

    def seekStream(self, ctx):
        self.core.seekFileStreamCall(ctx)

    def playStream(self, ctx):
        self.core.playFileStreamCall(ctx)

    def stopStream(self, ctx):
        self.core.stopFileStreamCall(ctx)

    def close(self):
        if not self.active:
            return
        if (self.verbose):
            print(self.pre, "stopping core.FileThread")
        self.core.stopCall()
        self.active = False

    def requestClose(self):
        self.core.requestStopCall()
        
    def waitClose(self):
        self.core.waitStopCall()
        self.active = False

    # """
    def __del__(self):
        self.close()
    # """


class OpenGLThread:
    # OpenGLThread(const char* name, unsigned short n720p=0, unsigned short
    # n1080p=0, unsigned short n1440p=0, unsigned short n4K=0, unsigned
    # msbuftime=100, int core_id=-1);

    parameter_defs = {
        "name": (str, "gl_thread"),
        "n_720p": (int, 20),
        "n_1080p": (int, 20),
        "n_1440p": (int, 0),
        "n_4K": (int, 0),
        "n_setup": (int, 20),
        "n_signal": (int, 20),
        "flush_when_full": (bool, False),
        "msbuftime": (int, 100),
        "affinity": (int, -1),
        "verbose": (bool, False),
        "background": (str, getDataFile("valkka_bw_logo.yuv")),
        "x_connection": (str, "")
    }

    def __init__(self, **kwargs):
        # auxiliary string for debugging output
        self.pre = self.__class__.__name__ + " : "
        # check kwargs agains parameter_defs, attach ok'd parameters to this
        # object as attributes
        parameterInitCheck(self.parameter_defs, kwargs, self)

        # some "api-level 1" objects here = swig-wrapped cpp objects
        #
        # This parameter set is defined at the cpp header file
        # "openglframefifo.h"
        self.gl_ctx = core.OpenGLFrameFifoContext()
        self.gl_ctx.n_720p = self.n_720p
        self.gl_ctx.n_1080p = self.n_1080p
        self.gl_ctx.n_1440p = self.n_1440p
        self.gl_ctx.n_4K = self.n_4K
        self.gl_ctx.n_setup = self.n_setup
        self.gl_ctx.n_signal = self.n_signal
        self.gl_ctx.flush_when_full = self.flush_when_full

        # OpenGLThread(const char* name, OpenGLFrameFifoContext
        # fifo_ctx=OpenGLFrameFifoContext(), unsigned
        # msbuftime=DEFAULT_OPENGLTHREAD_BUFFERING_TIME, const char*
        # x_connection="");
        self.core = core.OpenGLThread(
            self.name, self.gl_ctx, self.msbuftime, self.x_connection)
        self.core.setAffinity(self.affinity)
        self.core.setStaticTexFile(self.background)

        self.input_filter = self.core.getFrameFilter()

        self.render_groups = []  # list of registered x windows
        self.render_counts = []  # how many slots are mapped per each x window
        self.tokens = []  # unique tokens identifying the slot => x window mappings
        self.token_to_group = []  # gives group corresponding to each token by token index

        self.active = True
        if (self.verbose):
            print(self.pre, "starting core.OpenGLThread")
        self.core.startCall()
        if (self.verbose):
            print(self.pre, "started core.OpenGLThread")

    def getInput(self):
        return self.input_filter

    def close(self):
        if not self.active:
            return
        if (self.verbose):
            print(self.pre, "stopping core.OpenGLThread")
        self.core.stopCall()
        self.active = False

    def requestClose(self):
        self.core.requestStopCall()
        
    def waitClose(self):
        self.core.waitStopCall()
        self.active = False

    def __del__(self):
        self.close()

    def createWindow(self, show=True):
        """Create an x window.  For debugging/testing only.  Returns x-window id.
        """
        window_id = self.core.createWindow(show)
        self.core.makeCurrent(window_id)
        return window_id

    def newRenderGroup(self, window_id):
        return self.core.newRenderGroupCall(window_id)

    def delRenderGroup(self, window_id):
        return self.core.delRenderGroupCall(window_id)

    def newRenderContextCall(self, slot, window_id, z=0): # TODO: deprecated
        return self.core.newRenderContextCall(
            slot, window_id, z)  # returns token
    
    def newRenderContext(self, slot, window_id, z=0):
        return self.core.newRenderContextCall(
            slot, window_id, z)  # returns token
    
    def delRenderContext(self, token):
        return self.core.delRenderContextCall(token)

    def connect(self, slot=0, window_id=0):
        """Returns a unique token identifying the slot => x-window mapping.  Returns 0 if mapping failed.
        TODO: self.core (i.e. the cpp wrapped OpenGLThread already does accounting.. do we really need double accounting of render groups / contexes here ..?
        """
        if (slot == 0):
            raise(AssertionError("valid slot number missing"))

        if (window_id == 0):
            raise(AssertionError("valid window_id missing"))

        if (window_id not in self.render_groups):
            if (self.verbose):
                print(self.pre, "connect : new render group :", window_id)
            ok = self.core.newRenderGroupCall(window_id)
            if (ok):
                self.render_groups.append(window_id)
                self.render_counts.append(0)
            else:
                print(
                    self.pre,
                    "connect : WARNING : creating render group failed :",
                    window_id)
                return 0

        i = self.render_groups.index(window_id)
        self.render_counts[i] += 1

        # slot, window_id, z (not functional at the moment)
        token = self.core.newRenderContextCall(slot, window_id, 0)
        if (token == 0):
            print(self.pre, "connect : WARNING : creating render contex failed")
            # TODO: check if the render group should be eliminated..?
            return 0
        self.tokens.append(token)
        self.token_to_group.append(window_id)
        return token

    def disconnect(self, token=0):
        if (token == 0):
            raise(AssertionError("invalid token 0"))

        if (token not in self.tokens):
            print(self.pre, "disconnect : WARNING : no such token :", token)
        else:
            if (self.verbose):
                print(self.pre, "disconnect : removing token :", token)
            ok = self.core.delRenderContextCall(token)
            if (not ok):
                print(
                    self.pre,
                    "disconnect : WARNING : could not remove token :",
                    token)

            # get the index
            i = self.tokens.index(token)
            window_id = self.token_to_group[i]

            # remove token from the books
            self.tokens.pop(i)
            self.token_to_group.pop(i)

            i = self.render_groups.index(window_id)
            self.render_counts[i] -= 1
            if (self.render_counts[i] <=
                    0):  # nothing to map into this x window anymore
                if (self.verbose):
                    print(
                        self.pre,
                        "disconnect : removing render group :",
                        window_id)
                ok = self.core.delRenderGroupCall(window_id)
                if (not ok):
                    print(
                        self.pre,
                        "disconnect : WARNING : could not remove render group :",
                        window_id)
                self.render_groups.pop(i)  # remove from the books

    def hadVsync(self):
        return self.core.getVsyncAtStartup() > 0


def test1():
    st = """Test LiveThread encapsulation
  """
    import time

    pre = pre_mod + "test1 :"
    print(pre, st)

    livethread = LiveThread(
        name="live_thread"
    )

    print(pre, "sleepin'")
    time.sleep(3)
    print(pre, "bye!")

    # livethread.close() # TODO: should thread destructor stop the thread if
    # not already stopped?


def test2():
    st = """Test OpenGLThread encapsulation
  """
    pre = pre_mod + "test2 :"
    print(pre, st)

    openglthread = OpenGLThread(
        name="mythread",
        n_1440p=5,
        verbose=True
    )

    win_id = openglthread.createWindow()
    win_id1 = openglthread.createWindow()
    win_id2 = openglthread.createWindow()

    token = openglthread.connect(1, win_id)
    print(pre, "got token", token)
    openglthread.disconnect(token)
    print(pre, "disconnected token", token)

    print(pre, "connect token")
    token1 = openglthread.connect(1, win_id1)
    print(pre, "connect token")
    token11 = openglthread.connect(2, win_id1)
    print(pre, "connect token")
    token2 = openglthread.connect(3, win_id2)

    print(pre, "disconnect token", token1)
    openglthread.disconnect(token1)
    print(pre, "disconnect AGAIN token", token1)
    openglthread.disconnect(token1)
    print(pre, "disconnect token", token11)
    openglthread.disconnect(token11)
    print(pre, "disconnect token", token2)
    openglthread.disconnect(token2)

    # openglthread.close()


def test3():
    st = """Test OpenGLThread encapsulation again
  """
    pre = pre_mod + "test3 :"
    print(pre, st)

    openglthread = OpenGLThread(
        name="mythread",
        n_1440p=5,
        verbose=True
    )


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
