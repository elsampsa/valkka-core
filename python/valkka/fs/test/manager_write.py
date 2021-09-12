import time, sys
from valkka.core import *
from valkka.fs import ValkkaMultiFS, ValkkaFSManager

"""ValkkaFSManager write-only test

Filterchain

::
                                        
                  +---> ValkkaFSManager 
                  |                        
    LiveThread ---+                  
      (slot=1)         

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

"""Map stream in the manager

- writer & reader threads associated to vfs1 will map slot 1 to id 1
(- reader dumps to manager.cacherthread)
- cacherthread dumps slot 1 to framefilter
"""
info_filter = InfoFrameFilter("info")
manager.map_(
    valkkafs=vfs1,
    framefilter=info_filter,
    write_slot=1,
    read_slot=10,
    _id=1001
)
file_input_framefilter = manager.getInputFilter(vfs1)
ctx = LiveConnectionContext(LiveConnectionType_rtsp, rtsp_adr, 1, file_input_framefilter)

livethread = LiveThread("livethread")
manager.start() # starts cacherthread, reader & writer threads
livethread.startCall()
livethread.registerStreamCall(ctx)
livethread.playStreamCall(ctx)

# stream for 10 secs
time.sleep(10)

print("time to exit!")
manager.unmap(valkkafs = vfs1, _id=1001)

livethread.stopCall()
manager.close()
print("bye")
