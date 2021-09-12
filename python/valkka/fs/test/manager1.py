import time, sys
from valkka.core import *
from valkka.fs import ValkkaMultiFS, ValkkaFSManager

"""Filterchain

::


    LiveThread ---> ValkkaFSManager ---> AVThread --> OpenGLThread

"""

# rtsp_adr = "rtsp://admin:12345@192.168.0.157"
rtsp_adr = sys.argv[0]

"""Define ValkkaFS
"""
vfs1 = ValkkaMultiFS.newFromDirectory(
    dirname = "./vfs1",
    blocksize = 512*1024,
    n_blocks = 10,
    verbose = True)

manager = ValkkaFSManager([vfs1])

def time_cb(mstime):
    print("time_cb:", mstime)

def time_limits_cb(tup):
    print("time_limits_cb:", tup)

def block_cb(valkkafs = None, timerange_local = None, timerange_global = None):
    print("block_cb:", valkkafs, timerange_local, timerange_global)


"""ValkkaFSManager API
"""
tr = manager.getTimeRange()
tr_ = manager.getTimeRangeByValkkaFS(vfs1)
print("timeranges:", tr, tr_)
manager.setTimeCallback(time_cb)
manager.setTimeLimitsCallback(time_limits_cb)
manager.setBlockCallback(block_cb)

glthread = OpenGLThread("glthread")
gl_in_filter = glthread.getFrameFilter()
file_input_framefilter = manager.getInputFilter(vfs1)
avthread = AVThread("avthread",gl_in_filter)
av_in_filter = avthread.getFrameFilter()

"""Map stream in the manager

- writer & reader threads associated to vfs1 will map slot 1 to id 1
(- reader dumps to manager.cacherthread)
- cacherthread dumps slot 1 to framefilter
"""
manager.map_(
    valkkafs=vfs1,
    framefilter=av_in_filter,
    slot=1,
    _id=1
)
file_input_framefilter = manager.getInputFilter(vfs1)
livethread = LiveThread("livethread")
ctx = LiveConnectionContext(LiveConnectionType_rtsp, rtsp_adr, 1, file_input_framefilter)

glthread.startCall()
avthread.startCall()
manager.start() # starts cacherthread, reader & writer threads
livethread.startCall()

avthread.decodingOnCall()
livethread.registerStreamCall(ctx)
livethread.playStreamCall(ctx)

window_id =glthread.createWindow()
glthread.newRenderGroupCall(window_id)
context_id=glthread.newRenderContextCall(1, window_id, 0) # slot, render group, z

# stream for 30 secs
time.sleep(10)

print("time to exit!")
glthread.delRenderContextCall(context_id)
glthread.delRenderGroupCall(window_id)
# stop decoding
avthread.decodingOffCall()
manager.unmap(valkkafs = vfs1, slot = 1)

"""<rtf>
Close threads.  Stop threads in beginning-to-end order (i.e., following the filtergraph from left to right).
<rtf>"""
livethread.stopCall()
manager.close()
avthread.stopCall()
glthread.stopCall()

print("bye")
