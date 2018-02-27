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
print("test: LiveConnectionContext.request_multicast =",ctx.request_multicast)
ctx.slot=1
ctx.connection_type=LiveConnectionType_rtsp
ctx.address="rtsp://admin:12345@192.168.0.157"

ctx2=LiveConnectionContext(LiveConnectionType_rtsp, "rtsp://admin:12345@192.168.0.157", 1, out)
print("test: LiveConnectionContext.address =",ctx2.address)
