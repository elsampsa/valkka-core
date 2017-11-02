from valkka.valkka_core import *

"""Just test instatiation
"""
print("Version ",VERSION_MAJOR,".",VERSION_MINOR,".",VERSION_PATCH)

live =LiveThread("live")
inp  =FrameFifo("fifo",10)
ff   =FifoFrameFilter("fifo",inp)
out  =DummyFrameFilter("dummy")
av   =AVThread("av",inp,out)
gl   =OpenGLThread("gl")

ctx=LiveConnectionContext()
ctx.slot=1
ctx.connection_type=LiveConnectionType_rtsp
ctx.address="rtsp://admin:12345@192.168.0.157"

# ctx=LiveConnectionContext(slot=1,connection_type=LiveConnectionType_rtsp,address="rtsp://admin:12345@192.168.0.157") # we should create things like this in the level 2 api ..
