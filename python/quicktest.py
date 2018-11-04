from valkka.core import *

"""Just test instatiation
"""
print("Version ",VERSION_MAJOR,".",VERSION_MINOR,".",VERSION_PATCH)

live =LiveThread("live")
inp  =FrameFifo("fifo")
ff   =FifoFrameFilter("fifo",inp)
out  =DummyFrameFilter("dummy")
av   =AVThread("av",out)
gl   =OpenGLThread("gl")

av_in =av.getFrameFilter();
gl_in =gl.getFrameFilter();

ctx=LiveConnectionContext()
print("test: LiveConnectionContext.request_multicast =",ctx.request_multicast)
ctx.slot=1
ctx.connection_type=LiveConnectionType_rtsp
ctx.address="rtsp://admin:12345@192.168.0.157"

ctx2=LiveConnectionContext(LiveConnectionType_rtsp, "rtsp://admin:12345@192.168.0.157", 1, out)
print("test: LiveConnectionContext.address =",ctx2.address)
