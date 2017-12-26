"""
Api level 2 teaser:
  * Connect to an rtsp camera
  * Decode the stream once, and only once from H264 into YUV
  * Redirect the YUV stream into two branches:
  * Branch 1 goes into GPU, where YUV => RGB interpolation is done 25-30 fps.  The final bitmap is shown in multiple windows
  * Branch 2 is interpolated from YUV => RGB once in a second on the CPU and into a small size.  This small sized image is then passed over shared memory to opencv running in python.
"""

import sys
import time
import cv2
from valkka.api2.threads import LiveThread, OpenGLThread, ShmemClient
from valkka.api2.chains import BasicFilterchain, ShmemFilterchain

address="rtsp://admin:12345@192.168.0.157"

livethread=LiveThread(         # starts live stream services (using live555)
  name   ="live_thread",
  verbose=False
  )

openglthread=OpenGLThread(     # starts frame presenting services
  name    ="mythread",
  n720p   =10,  # reserve stacks of YUV video frames for various resolutions
  n1080p  =10,
  n1440p  =0,
  n4K     =0,
  naudio  =10,
  verbose =False
  )

# now livethread and openglthread are running

chain=ShmemFilterchain(       # decoding and branching the stream happens here
  livethread  =livethread, 
  openglthread=openglthread,
  address     =address,
  slot        =1,
  # this filterchain creates a shared memory server
  shmem_name             ="testing",
  shmem_image_dimensions =(1920//4,1080//4),  # Images passed over shmem are quarter of the full-hd reso
  shmem_image_interval   =1000,               # YUV => RGB interpolation to the small size is done each 1000 milliseconds and passed on to the shmem ringbuffer
  shmem_ringbuffer_size  =10                  # Size of the shmem ringbuffer
  )

# Let's create some x windows
win_id1 =openglthread.createWindow()
win_id2 =openglthread.createWindow()
win_id3 =openglthread.createWindow()

# Map video stream to three windowses
token1  =openglthread.connect(slot=1,window_id=win_id1) # map slot 1 to win_id1.  Frames are interpolated from YUV to RGB at the GPU
token2  =openglthread.connect(slot=1,window_id=win_id2)
token3  =openglthread.connect(slot=1,window_id=win_id3)
  
name, n_buffer, n_bytes =chain.getShmemPars()
print("name, n_buffer, n_bytes",name,n_buffer,n_bytes)

# let's create a shared memory client
client=ShmemClient(
  name          =name,       # e.g. "testing"
  n_ringbuffer  =n_buffer,   # 10
  n_bytes       =n_bytes,    # size of the RGB image
  mstimeout     =1000,       # client timeouts if nothing has been received in 1000 milliseconds
  verbose       =False
  )
  
chain.decodingOn() # tell the decoding thread to start its job
  
# All the threads are running at the c-level, so there is no GIL problems here.. now, let's start doing stuff in python:
t=time.time()
while True:
  index, isize = client.pull()
  if (index==None):
    print("Client timed out..")
  else:
    print("Client index, size =",index, isize)
    data=client.shmem_list[index]
    # print(">>>",data[0:10])
    img=data.reshape((1080//4,1920//4,3))
    # so, here we just dump the image once more - but I guess you got the idea
    cv2.imshow("openCV_window",img)
    cv2.waitKey(1)
  if ( (time.time()-t)>=20 ): break # exit after 20 secs
  
# that ShmemClient could be instantiated from a forked or even an independent python process, and this would still work as long as you
# use same name for ShmemClient: name and ShmemFilterchain: shmem_name.  It's named and shared posix memory and semaphores.

openglthread.disconnect(token1)
openglthread.disconnect(token2)
openglthread.disconnect(token3)
  
# garbage collection takes care of stopping threads etc.

