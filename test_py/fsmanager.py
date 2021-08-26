import os, time
from valkka import core  # for logging
from valkka.api2.logging import *
from valkka.api2 import LiveThread,\
    OpenGLThread, ValkkaFS, ValkkaFSManager,\
    ValkkaFSLiveFilterchain, ValkkaFSFileFilterchain

# adr = "rtsp://admin:12345@192.168.0.157"
adr = "rtsp://admin:123456@10.0.0.4"
blocksize = 1*1024*1024 # 1 MB
device_size = 1024*1024*1024*100 # 100 GB
partition_uuid = "37c591e3-b33b-4548-a1eb-81add9da8a58"
valkka_fs_dirname = "./testvalkkafs"
"""
try:
    ValkkaFS.checkDirectory(valkka_fs_dirname)
except Exception as e:
    print("Can't init ValkkaFileSystem.  Consider removing directory %s" %
          (valkka_fs_dirname))
    raise(e)
else:
"""
assert(os.path.exists(valkka_fs_dirname))
print("creating ValkkaFS for %i MBytes", blocksize/1024/1024)
valkkafs = ValkkaFS.newFromDirectory(
    dirname=valkka_fs_dirname,
    partition_uuid=partition_uuid,
    blocksize=blocksize,
    device_size=device_size,
    verbose=True
)

valkkafsmanager = ValkkaFSManager(
    valkkafs,
    # read = False,
    # cache = False,
    # write = False
)

input_framefilter = valkkafsmanager.getInputFrameFilter()
valkkafsmanager.setInput(999, 1) # _id, slot
# valkkafsmanager.setOutput(999, 1, ff) # _id, slot, framefilter
# TODO: ff => OpenGLThread

livethread = core.LiveThread("livethread")
livethread.startCall()
ctx = core.LiveConnectionContext(core.LiveConnectionType_rtsp, adr, 1, input_framefilter)
livethread.registerStreamCall(ctx)
livethread.playStreamCall(ctx)

print("sleepin 10 secs")
time.sleep(10)

print("Blocktable peek")
a = valkkafs.getBlockTable()
print(a[:, 0:10])

print("sleepin 10 secs")
time.sleep(10)

print("Blocktable peek")
a = valkkafs.getBlockTable()
print(a[:, 0:10])

#TODO: do operations with ValkkaFSManager

# stop all threads
livethread.stopCall()
valkkafsmanager.close()

"""Create tests for:

- Several cameras
- Keep streaming into the FS while playing, seeking, etc.
"""
