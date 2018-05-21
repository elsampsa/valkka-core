# Valkka - OpenSource Video Surveillance and Management for Linux

## Synopsis

The goal of this project is to provide a library for creating open source video surveillance, management and analysis systems (VMAs) in Linux environment.  The idea is create VMA systems with graphical user interfaces (GUIs) using the combination of python3 and Qt (i.e. PyQt).

## For the impatient

Installation instructions, demo programs and API tutorial are available [here](https://elsampsa.github.io/valkka-examples/) (you should read that first)

If you just want to use the API, no need to go further.

If you are interested in the core development, keep on reading.

## Why this library?

Valkka is *not* your browser-based-javacsript-node.js-cloud toy.  We're writing a library for building large ip camera systems in LAN (or virtual-LAN / VPS) environments, capable of doing simultaneously massive live video streaming, surveillance, recording and machine vision.

Lets take a look at a typical video management system architecture problem:
- Stream H264 video from an IP camera using the RTSP protocol
- Branch that stream, and direct it to (1) filesystem and (2) a decoder
- From the decoder, branch the decoded YUV bitmap to (3-5):
  - (3) Analyzer process, using python OpenCV, that inspects the video once per second for car license plates
  - (4) A Fullscreen X-window on screen 1
  - (5) To a smaller X-window on screen 2
- The media stream should be decoded once and only once
- Graphical interface should be based on a solid GUI desktop framework (say, Qt or GTK)

You might try to tackle this with some available stock media player libraries, but I'll promise, you wont get far.

Consider further that in a typical VMA system you may have up to 60+ ip cameras plugged into the same server.  Servers should also work as a proxies, re-streaming the ip cameras to other servers.

Using Valkka, you can instantiate threads, and define how media streams are branched and pipelined between those threads.  The underlying threads and mutex-protected queues are hidden from the developer that controls everything using a python3 API.  The process topology of the example case would look like this:


    [LiveThread]->|                        +-----> [AnalyzerProcess]
                  |                        | (branch 1)
                  +--> [DecoderThread] --->| 
                  |                        | (branch 2)  
                  +--> Filesystem          +------> [OpenGLThread] -- > X window system
             
Some key features of the Valkka library are:
- Python3 API: create process topologies from Python3.
- Develop sleek graphical interfaces fast with PyQt.
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
- For an overview of technical details, see [documentation](https://elsampsa.github.io/valkka-core/).  If you are just using the python3 API, you should read at least the "Library Architecture" section.


## Versions and Features

We're currently at alpha

### Current stable version is 0.4.3

0.4.3 Version
- Now reads acc in addition to annex b h264

### Older versions

0.4.0 Version : "The Rewrite"
- A complete rewrite of the library
- (documentation might lag a bit behind..)

0.3.6 Version
- Sending multicast works

0.3.5 Version : "10 x 1080p cameras running for a week"
- Stable!  GPU direct memory access needed rubbing in the right way
- Lots of fixes ..
- Reading / writing from/to matroska container

0.3.0 Version name : "It was all about the vblank"
- Several full-HD cameras now streaming OK
- Interoperability with python multiprocesses (and from there, with OpenCV)
- For benchmarking, testing and demos see the "valkka-examples" repository

0.2.1 Version
- License change (to APGL)
- Added python level 2 api example
- Miscellaneous fixes

0.2.0 Version name : "Christmas 2017 project"
- Software interpolator filter (yuv => rgb interpolation in the CPU)
- Shared memory bridge for python inter-process communication
- Python level 2 api
- Just committed this one : documentation and packages will be updated soon :)

0.1.0 Version name : "Proof of concept"
- Initial git commit: core system, live streaming to X-window system

### Long term goals
- Interserver communication and stream proxying
- ValkkaFS filesystem, saving and searching video stream
- A separate python3 Onvif module

### Very long term goals
- A complete video management & analysis system

## Compile yourself

### Dependencies

You need (at least):

    sudo apt-get install yasm git swig python3-pip cmake libx11-dev libglew-dev libasound2-dev pkg-config

Install also the following python packages:
    
    pip3 install --upgrade ipython numpy 
    
### Compile

You have two options (A, B):

A. Using your custom-compiled / home-brewn Live555 and FFmpeg.  This is recommended.
  - Download and compile Live555 and FFmpeg libraries
  - The script "lib/run_config_3_4_.bash" will help you to compile a stripped-down version of FFmpeg 3.4 libraries
  - Read the comments in "lib/ln_live.bash" and "lib/ln_ffmpeg.bash"
  - Use those scripts to create softlinks to the library files
  - NEW: script "aux/valkka_builder.bash" has it all (however, you should not launch it before reading/studying it).

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

## Resources

1. Discussion threads:

  - [GLX / OpenGL Rendering](https://www.opengl.org/discussion_boards/showthread.php/200394-glxSwapBuffers-and-glxMakeCurrent-when-streaming-to-multiple-X-windowses)

2. Doxygen generated [documentation](https://elsampsa.github.io/valkka-core/)
3. The examples [repository](https://github.com/elsampsa/valkka-examples)

## Authors
Sampsa Riikonen (core programming, opengl shader programming, python programming)

Petri Eranko (financing, testing)

Marco Eranko (testing)

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
