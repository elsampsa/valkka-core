# Valkka - OpenSource Video Surveillance and Management for Linux

## Synopsis

The goal of this project is to provide a library for creating open source video surveillance, management and analysis systems (VMAs) in Linux environment.  The idea is create VMA systems with graphical user interfaces (GUIs) using the combination of python3 and Qt (i.e. PyQt).

## For the impatient

For a demo program that uses libValkka, see [Valkka Live.](https://elsampsa.github.io/valkka-live/)

Installation instructions, demo programs and API tutorial are available [here](https://elsampsa.github.io/valkka-examples/) (you should read that first)

If you just want to use the API, no need to go further.

If you are interested in compiling Valkka yourself or even help us with the core development, keep on reading.

## Why this library?

Valkka is *not* your browser-based-javacsript-node.js-cloud toy.  We're writing a library for building large ip camera systems in LAN (or virtual-LAN / VPN) environments, capable of doing simultaneously massive live video streaming, surveillance, recording and machine vision.

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
- For more technical details, see [documentation](https://elsampsa.github.io/valkka-core/).  If you are just using the python3 API, you should read at least the "Library Architecture" section.


## Versions and Features

We're currently at alpha

### Current stable version is 0.9.0

0.9.0 Version
- H264 USB Cameras work

### Older versions

0.8.0 Version : "Say Yolo again"
- Added bounding boxes to OpenGLThread API
- Tested Valkka Live with YOLOv3 object detection
- Started sketching USB camera thread and ValkkaFS

0.7.1 Version
- Small fix in the python part valkka.api2 (in managed filterchain)

0.7.0 Version : "One namespace to rule them all"
- Switched to python namespace packaging

0.6.0 Version
- etc

0.5.4 Version
- Python API-level 2 : A managed filterchain that handles streams on-demand between processes

0.5.3 Version
- Woops .. there was no regression but a nasty bug with the use of stl containers
- .. works fine now, after I followed the "rule of four" (ctor, dtor, copy-ctor and copy-assignment)

0.5.2 Version
- Fixed a small bug: smart timestamp correction is again the default
- Weird regression here: segfault with intel gfx drivers with 5+ streams.  Problem with the latest intel driver?

0.5.1 Version : "Videowalls"
- Multi-GPU works

0.5.0 Version : "Halfway to Beta"
- Live rendering very solid

0.4.7 Version
- Can change receiving socket size and Live555 reordering buffer
- etc.

0.4.6 Version
- Background texture when no stream is coming through
- Weird bitmap sizes work
- TestThread class for sending (PyQt) signals from cpp
- etc.

0.4.4 Version
- RTSP server works

0.4.3 Version
- Now reads acc in addition to annex b h264

0.4.0 Version : "The Rewrite"
- A complete rewrite of the library
- (documentation might lag a bit behind..)

0.3.6 Version
- Sending multicast works

0.3.5 Version : "10 x 1080p cameras running for a week"
- Stable!  GPU direct memory access needed rubbing in the right way (..or did it?)
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

## Installing

Binary packages and their Python3 bindings are provided for latest Ubuntu distributions.  Subscribe to our repository with: 

    sudo apt-add-repository ppa:sampsa-riikonen/valkka
    
and then do:
    
    sudo apt-get update
    sudo apt-get install valkka

## Compile yourself

### Dependencies

You need (at least):

    sudo apt-get install build-essential libc6-dev yasm cmake pkg-config swig libglew-dev mesa-common-dev libstdc++-5-dev python3-dev python3-numpy libasound2-dev
    
### Compile

This just got a lot easier: the same CMake file is used to compile the library, generate python wrappings and to compile the wrappings (no more python setup scripts)

Valkka uses numerical python (numpy) C API and needs the numpy C headers at the build process.  Be aware of the numpy version and header files being used in your setup.  You can check this with:

    ./pythoncheck.bash
    
We recommend that you use a "globally" installed numpy (from the debian *python3-numpy* package) instead of a "locally" installed one (installed with pip3 install).  When using your compiled Valkka distribution, the numpy version you're loading at runtime must match the version that was used at the build time.

Next, download live555 and ffmpeg

    cd ext
    ./download_live.bash
    ./download_ffmpeg.bash
    cd ..

Next, proceed in building live555, ffmpeg and valkka 
    
    make -f debian/rules clean
    make -f debian/rules build
    
Finally, create a debian package with

    make -f debian/rules package
    
You can install the package to your system with

    cd build_dir
    dpkg -i Valkka-*.deb
    sudo apt-get -fy install
    
### Development environment
    
If you need more fine-grained control over the build process, create a separate build directory and copy the contents of the directory *tools/build* there.  Read and edit *run_cmake.bash* and *README_BUILD*.  Now you can toggle various debug/verbosity switches, define custom location for live555 and ffmpeg, etc.  After creating the custom build, you should run

    source test_env.bash

in your custom build directory.  You still need to inform the python interpreter about the location of the bindings.  In the main valkka directory, do:

    cd python
    source test_env.bash
    
And you're all set.  Now you have a terminal that finds both libValkka and the python3 bindings

### Semi-automated testing

After having set up your development environment, made changes to the code and succesfully built Valkka, you should run the testsuite.  Valkka is tested by a series of small executables that are using the library, running under valgrind.  For some of the tests, valgrind can't be used, due to the GPU direct memory access.  For these tests, you should (i) run them without valgrind and see if you get video on-screen or (ii) compile valkka with the VALGRIND_DEBUG switch enabled and only after that, run them with valgrind.

In your build directory, refer to the bash script *run_tests.bash*.  Its self-explanatory.

Before running *run_tests.bash" you should edit and run the *set_test_streams.bash* that sets up your test cameras.


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
