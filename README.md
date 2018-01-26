# Valkka - Massive video streaming for linux

## Synopsis
The goal of this project is to provide a library for creating open source video surveillance, management and analysis systems (VSMAs) in Linux environment.  The idea is to be able create VSMA systems with graphical user interfaces (GUIs) using the combination of python3 and Qt (i.e. PyQt).

Lets take a look at a typical VSMA programming architecture problem:
- Stream H264 video from an IP camera using the RTSP protocol
- Branch that stream, and direct it to (1) filesystem and (2) a decoder
- From the decoder, branch the decoded YUV bitmap to (3-5):
  - (3) Analyzer process, using python OpenCV, that inspects the video once per second for car license plates
  - (4) A Fullscreen X-window on screen 1
  - (5) To a smaller X-window on screen 2
- The media stream should be decoded once and only once
- The program should be controllable through python3, with graphical interface based on Qt

You might try to tackle this with some available stock media player libraries, but I'll promise, you wont get far..!

Consider further that in a typical VSMA system you may have up to 60+ ip cameras plugged into the same server.  Servers should also work as a proxies, re-streaming the ip cameras to other servers.

Using Valkka, you can instantiate threads, and define how media streams are branched and pipelined between those threads.  The underlying threads and mutex-protected queues are hidden from the developer that controls everything using a python3 API.  The process topology of the example case would look like this:


    [LiveThread]->|                        +-----> [AnalyzerProcess]
                  |                        | (branch 1)
                  +--> [DecoderThread] --->| 
                  |                        | (branch 2)  
                  +--> Filesystem          +------> [OpenGLThread] -- > X window system

Check out the API teaser in the end of this document.
             
Some key features of the Valkka library are:
- Python3 API: create process topologies from python3 only.
- Develop sleek graphical interfaces fast with PyQt.  Cool!
- The library itself runs purely in C++.  No python Global Interpreter Lock (GIL) problems here.
- Connections to streaming devices (IP cameras, SDP files) are done using the Live555 media streaming library
- Decoding is done with the FFMpeg library
- Asynchronous texture uploading to GPU using OpenGL.
- Bitmap interpolations with the OpenGL shader language (i.e. GPU does some of the effort)
- The programming architecture makes it possible to implement (some more core development needed):
  - Composite "video in video" (think skype) images
  - Arbitrary geometry transformations : think of fisheye spheres, etc.
  - And much more .. !
- Two-level API.  Level 1 is simply swig-wrapped cpp.  Level 2 is higher level and makes development even easier.


See also the list of (latest) features below.

## For the impatient
- You need to install two pre-built packages from [here](https://www.dropbox.com/sh/cx3uutbavp2cqpa/AAC_uDh-plu0Oo50r_klYPEXa?dl=0)
- Install the debian (.deb) package with: 

      sudo dpkg -i package_name
      sudo apt-get -f install
    
- Install python3 binary package (.whl) with: 

      pip3 install --upgrade package_name

- Download python3 examples from "valkka-examples" [repository](https://github.com/elsampsa/valkka-examples).
- Check out Valkka cpp [documentation](https://elsampsa.github.io/valkka-core/).  If you are just using the python3 API, you should read at least the "Library Architecture" section.

## Features

### Current stable version is 0.2.1
- Full-HD / 25 fps with several cameras streaming allright - but only when using X-windoses created with the library itself (see Resources)
- License change (to APGL)
- Added python level 2 api example
- Miscellaneous fixes

### Older versions
0.2.0 Version name : "Christmas 2017 project"
- Software interpolator filter (yuv => rgb interpolation in the CPU)
- Shared memory bridge for python inter-process communication
- Python level 2 api
- Just committed this one : documentation and packages will be updated soon :)

0.1.0 Version name : "Proof of concept"
- Initial git commit: core system, live streaming to X-window system

### Features coming soon
- Composite "video in video"
- Writing to matroska files
- Reading from matroska files
- Audio reproduction

### Long term goals
- Interserver communication and stream proxying
- ValkkaFS filesystem, saving and searching video stream
- A separate python3 Onvif module

### Very long term goals
- A complete VSMA system

## Compile and deploy

A word of warning: if you just want to use the API, no need to go further

However, if you have decided to develop Valkka and build it from source - great!  Here are the instructions:

### Dependencies

You need (at least):

    sudo apt-get install yasm git swig python3-pip cmake libx11-dev libglew-dev libasound2-dev pkg-config

Install also the following python packages:
    
    pip3 install --upgrade ipython numpy 
    
### Compile

You have two options (A, B):

A. Using your custom-compiled / home-brewn Live555 and FFmpeg.  This is recommended.
  - Download and compile Live555 and FFmpeg libraries (should not be under this directory structure, though)
  - The script "lib/run_config_.bash" will help you to compile a stripped-down version of FFmpeg
  - Read the comments in "lib/ln_live.bash" and "lib/ln_ffmpeg.bash"
  - Use those scripts to create softlinks to the library files

B. Using system-wide installed (i.e. with apt-get) shared libraries and header files
  - Just edit your run_cmake.bash accordingly (below)

Next, launch "./new_build.bash buildname", where buildname is a name you wish to tag your build with

Go to the newly created build directory and there:
  - Read "README_BUILD" and "README_TESTS"
  - Edit "run_cmake.bash"
  - Finally, run it with "./run_cmake.bash"
  - Do "source test_env.bash"
  - .. and now you should be able to run the test programs
  - Debian package can be created simply with "make package"

Creating the python3 interface: go to "python/" directory and read there "README.md"

### Contribute

Want to modify and develop further Valkka source code?  Your effort is needed.  You should start by reading some of the cpp documentation.

- Found a bug?  Send a message or a patch here or to Valkka [google group](https://groups.google.com/forum/#!forum/valkka)
- Want to implement a new feature?  Create your own development branch.  We will merge it once you have ran the tests (refer to "README_TESTS" in your build directory)

## Resources

1. Discussion threads

  -[GLX / OpenGL Rendering](https://www.opengl.org/discussion_boards/showthread.php/200394-glxSwapBuffers-and-glxMakeCurrent-when-streaming-to-multiple-X-windowses)

2. Doxygen generated [documentation](https://elsampsa.github.io/valkka-core/)
3. Google [group](https://groups.google.com/forum/#!forum/valkka)
4. The examples [repository](https://github.com/elsampsa/valkka-examples)

## Authors
Sampsa Riikonen (core programming, opengl shader programming, python programming)

Marco Eranko (testing)
Petri Eranko (financing, testing)

Markus Kaukonen (opengl shader programming, testing)

## Acknowledgements

Ross Finlayson
Dark Photon
GClements

## Copyright
(C) 2017, 2018 Valkka Security Ltd. and Sampsa Riikonen

## License
This software is licensed under the GNU Affero General Public License (AGPL) v3 or later.

If you need a different license arrangement, please contact us.

## Appendum.  API Teaser

Api level 2 teaser:

- Connect to an rtsp camera
- Decode the stream once, and only once from H264 into YUV
- Redirect the YUV stream into two branches:
- Branch 1 goes into GPU, where YUV => RGB interpolation is done 25-30 fps.  The final bitmap is shown in multiple windows
- Branch 2 is interpolated from YUV => RGB once in a second on the CPU and into a small size.  This small sized image is then passed over shared memory to opencv running in python.

Code:

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
        # so, here we just dump the image once more, this time with opencv - but I guess you got the idea
        cv2.imshow("openCV_window",img)
        cv2.waitKey(1)
      if ( (time.time()-t)>=20 ): break # exit after 20 secs
      
    # that ShmemClient could be instantiated from a forked or even an independent python process, and this would still work as long as you
    # use same name for ShmemClient: name and ShmemFilterchain: shmem_name.  It's named and shared posix memory and semaphores.

    openglthread.disconnect(token1)
    openglthread.disconnect(token2)
    openglthread.disconnect(token3)
      
    # garbage collection takes care of stopping threads etc.






