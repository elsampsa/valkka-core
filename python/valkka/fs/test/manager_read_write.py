import time, sys
from valkka.core import *
from valkka.fs import ValkkaMultiFS, ValkkaFSManager

"""Full read/write test/example using ValkkaFSManager

Filterchain

::
                                        (slot=10)
                  +---> ValkkaFSManager ---> AVThread ->-+
                  |                                      |
    LiveThread ---+ Fork                                 |
      (slot=1)    |                                      |
                  +---> AVThread --------------------->--+--> OpenGLThread

"""

# rtsp_adr = "rtsp://admin:12345@192.168.0.157"
rtsp_adr = sys.argv[1]

print("camera:", rtsp_adr)

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

live_av_thread = AVThread("live_avthread",gl_in_filter)
live_in_filter = live_av_thread.getFrameFilter()

playback_av_thread = AVThread("playback_avthread",gl_in_filter)
playback_in_filter = playback_av_thread.getFrameFilter()

"""Map stream in the manager

- writer & reader threads associated to vfs1 will map slot 1 to id 1
(- reader dumps to manager.cacherthread)
- cacherthread dumps slot 1 to framefilter
"""
manager.map_(
    valkkafs=vfs1,
    framefilter=playback_in_filter,
    write_slot=1,
    read_slot=10,
    _id=1001
)
file_input_framefilter = manager.getInputFilter(vfs1)
livethread = LiveThread("livethread")

fork_filter = ForkFrameFilter("fork", file_input_framefilter, live_in_filter)

ctx = LiveConnectionContext(LiveConnectionType_rtsp, rtsp_adr, 1, fork_filter)

glthread.startCall()
live_av_thread.startCall()
playback_av_thread.startCall()
manager.start() # starts cacherthread, reader & writer threads
livethread.startCall()

live_av_thread.decodingOnCall()
playback_av_thread.decodingOnCall()
livethread.registerStreamCall(ctx)
livethread.playStreamCall(ctx)

# live video window
live_window_id =glthread.createWindow()
glthread.newRenderGroupCall(live_window_id)
live_context_id=glthread.newRenderContextCall(1, live_window_id, 0) # slot, render group, z

# playback video window
playback_window_id =glthread.createWindow()
glthread.newRenderGroupCall(playback_window_id)
live_context_id=glthread.newRenderContextCall(10, playback_window_id, 0) # slot, render group, z

# stream for 30 secs
time.sleep(10)

print("time to exit!")
glthread.delRenderContextCall(live_context_id)
glthread.delRenderGroupCall(live_window_id)
# stop decoding
live_av_thread.decodingOffCall()
playback_av_thread.decodingOffCall()
manager.unmap(valkkafs = vfs1, _id=1001)

"""<rtf>
Close threads.  Stop threads in beginning-to-end order (i.e., following the filtergraph from left to right).
<rtf>"""
livethread.stopCall()
manager.close()
live_av_thread.stopCall()
playback_av_thread.stopCall()
glthread.stopCall()

print("bye")
