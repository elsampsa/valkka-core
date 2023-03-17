# Valkka - OpenSource Video Surveillance and Management for Linux

## For the VERY impatient

Looking for an OpenSource video surveillance program with object detection?  Just go [here](https://elsampsa.github.io/valkka-live/).

## Synopsis

The goal of this project is to provide a library for creating open source video surveillance, management and analysis systems (VMAs) in Linux environment.  The idea is create VMA systems with graphical user interfaces (GUIs) using the combination of python3 and Qt (i.e. PyQt).

## For the impatient

For a demo program that uses libValkka, see [Valkka Live.](https://elsampsa.github.io/valkka-live/)

Installation instructions, demo programs and API tutorial are available [here](https://elsampsa.github.io/valkka-examples/) (you should read that first)

If you just want to use the API, no need to go further.

If you are interested in compiling Valkka yourself or even help us with the core development, keep on reading.

## Why this library?

Most of the people nowadays have a concentration span of milliseconds (because of mobile devices).  Thank you for keep on reading!  :)

Lets take a look at a typical video management system architecture problem.  This goes for desktop / Qt-based programs and for web-based apps as well:

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

### Newest version is 1.3.5

1.3.5

- In the python path, restructured valkka.multiprocess namespace package.  Now it is also avail as a [separate package](https://github.com/elsampsa/valkka-multiprocess).

### Older versions

1.3.4

- Live555 RTSP sessions that might hang now properly closed.  Filterchain is "cut" when a connection is deregistered -> LiveThread doesn't write frames to filterchain that is being garbage collected
- Added EventFd into the Python API

1.3.3

- Added dependency to glew-utils package which fixes library dependence problems for all recent ubuntu LTS distros

- A consistent docker testing scheme for build and install in [docker/](docker/)

1.3.1

- Removed ``sysctl.h`` depedency from ffmpeg ``config.h``

1.3.0

- Recording a single-stream per file is now functional (cpp class: ValkkaFS2)
- Lots of reorganization under valkka.fs namespace: FSGroup, ValkkaFSManager, etc.
- Corresponding changes done to valkka-examples & valkka-live (phew)

1.2.2

- Using ``glFinish`` in Intel graphics driver OpenGL completely clogged the frame presentation pipeline resulting in lots of dropped frames.  Removing ``glFinish`` fixed the issue.
- Reorganized the Python Qt examples

1.2.1

- valkka.multiprocess.base.AsyncBackMessageProcess.run fixed: a separate event loop is needed in the async multiprocess

1.2.0

- AVThread subclassing etc. rewritten to allow separate hw decoding modules
- Accelerated decoding as a separate extension module available [here](https://github.com/xiaxoxin2/valkka-nv)

1.0.3

- Added method ``waitReady`` for libValkka threads: it should be called in the python API at garbage collection, so that active framefilters are not garbage collected while (Live)Thread is still writing into them
- Fixed a small compatibility issue with the latest live555 version

1.0.2

- Subsession-index mess fixed for now: the one-and-only subsession index (as we only support video) is set to 0.
- Frag-mp4 muxer fixed: there were memleaks when (de/re)activating the muxer.
- Shmem server-side bug fixed: eventfds we're accidentally closed when shmem server was closed, resulting in mysterious bugs when recycling eventfds.
- Debian auto-build now hopefully works for arm-based architectures as well (tested in docker).

1.0.1

- libValkka is now LGPL, hooray!
- Frag-MP4 debugged

1.0.0

- Changed shmem API, more examples added to valkka-examples
- ..and Python GIL now released by default in shmem operations
- Frag-MP4 streaming implemented
- "define analyzer" window & qt bitmap python reference leak still persists..

For more, see [CHANGELOG](CHANGELOG.md)

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
```
sudo apt-get install python3 mesa-utils glew-utils python3-numpy v4l-utils python3-pip openssl build-essential yasm cmake pkg-config swig libglew-dev mesa-common-dev python3-dev python3-numpy libasound2-dev libssl-dev coreutils freeglut3-dev
```

If you have upgraded your python interpreter, you might need to define the version, say ```python3.7-dev```

### Compile

The same CMake file is used to compile the library, generate python wrappings and to compile the wrappings (no more python setup scripts)

Valkka uses numerical python (numpy) C API and needs the numpy C headers at the build process.  Be aware of the numpy version and header files being used in your setup.  You can check this with:

    ./pythoncheck.bash
    
We recommend that you use a "globally" installed numpy (from the debian *python3-numpy* package) instead of a "locally" installed one (installed with pip3 install).  When using your compiled Valkka distribution, the numpy version you're loading at runtime must match the version that was used at the build time.

First, download ffmpeg source code:

    cd ext
    ./download_ffmpeg.bash
    cd ..

Then, just

    ./easy_build.bash
    
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
Copyright (c) 2017-2020 Valkka Security Ltd. and Sampsa Riikonen

## Open Source Licenses

- [Live555](http://www.live555.com/) Copyright (c) Live Networks, Inc.  LGPL License.
- [FFMpeg](https://www.ffmpeg.org/) Copyright (c) The FFMpeg authors.  LGPL Licence.
- [WSDiscovery](https://github.com/andreikop/python-ws-discovery) Copyright (c) L. A. Fernando.  LGPL License.


## License
GNU Lesser General Public License v3 or later.

(if you need something else, please contact us)
