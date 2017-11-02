# Valkka - Massive video streaming for linux

## Synopsis
The goal of this project is to provide a library for creating open source video surveillance, management and analysis systems (VSMAs) in Linux environment.  The idea is to be able create VSMA systems with graphical user interfaces (GUIs) using the combination of Python3 and Qt (i.e. PyQt).

Lets take a look at a typical VSMA programming architecture problem:
- Stream H264 video from an IP camera using the RTSP protocol
- Branch that stream, and direct it to (1) filesystem and (2) a decoder
- From the decoder, branch the decoded YUV bitmap to (3-5):
  - (3) Analyzer process, using python OpenCV, that inspects the video once per second for car license plates
  - (4) A Fullscreen X-window on screen 1
  - (5) To a smaller X-window on screen 2
- The media stream should be decoded once and only once
- The program should be controllable through python, with graphical interface based on Qt

You might try to tackle this with some available Linux stock media players, but I will promise, you wont get far..!

Consider further that in a typical VSMA system you may have up to 60+ (depending on the server hardware) ip cameras plugged into the same server.  Servers should also work as a proxies, re-streaming the ip cameras to other servers.

Using Valkka, you can instantiate threads, and define how media streams are branched and pipelined between those threads.  The underlying threads and mutex-protected queues are hidden from the developer that controls everything using a Python3 API.  The process topology of the example case would look like this:


    [LiveThread] |
                 | --> [DecoderThread] -->  | --> [AnalyzerProcess] 
                 |                          |
                 | --> ValkkaFS             | --> [OpenGLThread] -- > X window system

             
Some key features of the Valkka library are:
- Python API: create process topologies from Python only.  
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

See also the list of (latest) features below.

## For the impatient
- Grab one of the following debian packages, and install it with "dpkg -i package_name".
- Install also the python package and install it with "pip3 install package_name"
- Download python examples from "valkka-examples" repository
- Check out Valkka cpp [documentation](https://elsampsa.github.io/valkka-core/).  If you are just using the python API, you should read at least the "Library Architecture" section.

## Features

### Current stable version is 0.1.0  
Version name : "Proof of concept"

### Latest new features
- Initial git commit: core system, live streaming to X-window system

### Features coming ASAP
- Software interpolator filter (yuv => rgb interpolation in the CPU)
- Shared memory bridge for python inter-process communication

### Features coming soon
- Composite "video in video"
- Writing to matroska files
- Reading from matroska files
- Audio reproduction

### Long term goals
- Interserver communication and stream proxying
- ValkkaFS filesystem, saving and searching video stream
- A separate Python3 Onvif module

### Very long term goals
- A complete VSMA system

## Compile, contribute and deploy

A word of warning: if you just want to use the API, no need to go further

However, if you have decided to develop Valkka and build it from source - great!  Here are the instructions:

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

Creating the python interface: go to "python/" directory and read there "README.md"

### Contribute

Want to modify and develop further Valkka source code?  Your effort is needed.  You should start by reading some of the cpp documentation.

- Found a bug?  Send a message or a patch here or to Valkka google group
- Want to implement a new feature?  Create your own development branch.  We will merge it once you have ran the tests (refer to "README_TESTS" in your build directory)

## Resources
1. Doxygen generated [documentation](https://elsampsa.github.io/valkka-core/)
2. Google group (coming up in a minute)
3. The examples repository (coming up in a minute)

## Authors
Sampsa Riikonen (core programming)

Petri Eranko (financing, testing)

Markus Kaukonen (opengl shader programming)

## Copyright
(C) 2017 Valkka Security Ltd. and Sampsa Riikonen

## License
Lesser General Public License v3 or later

