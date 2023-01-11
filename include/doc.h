/*
 * doc.h : Dummy header file for doxygen documentation
 * 
 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>
 *
 */

/** 
 *  @file    doc.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.3.4 
 *  
 *  @brief Extra doxygen documentation
 *
 */

/* Some nice links:
 * coding style:
 * http://csweb.cs.wfu.edu/~fulp/CSC112/codeStyle.html
 * doxygen:
 * http://flcwiki.desy.de/How%20to%20document%20your%20code%20using%20doxygen
 * https://www.stack.nl/~dimitri/doxygen/manual/
 * https://stackoverflow.com/questions/51667/best-tips-for-documenting-code-using-doxygen
 * https://stackoverflow.com/questions/2544782/doxygen-groups-and-modules-index
 * 
 */


/** @mainpage
 * 
 *
 * Valkka - OpenSource Video Management
 * ------------------------------------
 *
 * The long-term goal of this project is to provide open source video surveillance, management and analysis systems (VMAs) in Linux environment, and to be able to create them with the combination of Python3 and Qt (i.e. PyQt).  The library has both Cpp and Python3 API (the latter is recommended).
 *
 * Warning: If you simply want to use Valkka's python3 API, you should start from <https://elsampsa.github.io/valkka-examples/>.
 * 
 * Authors
 * -------
 * Sampsa Riikonen <sampsa.riikonen@iki.fi> (core programming) <br> 
 * Petri Eränkö <petri.eranko@dasys.fi> (financing, testing) <br>
 * Markus Kaukonen <markus.kaukonen@iki.fi> (OpenGL shaders) <br>
 * 
 * 
 * Contributing
 * ------------
 * If you wish to contribute, please, read \ref contributing
 * 
 * For library architecture, code walkthroughs and explanations, check out the "Related Pages" section
 * 
 * License
 * -------
 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
 * 
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>
 *
 */

/** @page contributing Contributing
 * 
 * Instructions for building Valkka can be found from the main Github page, please read it first
 * 
 * Here are some miscellaneous tips for developers:
 * 
 * - In your build directory, don't forget to read README_BUILD and README_TESTS
 * - Read and understand (at least some of) \ref process_chart
 * - In the header files ("include/*.h") you'll see text "<pyapi>" .. don't erase them.  They're used to autogenerate swig wrappers for python interface building.  Refer to "python/make_swig_file.bash".
 * - When creating your own python calls, add "<pyapi>" to relevant lines in the header files
 * - For creating a new .cpp or .h file, use the "create.bash" script present both in "src/" and "include/"
 * - Before starting to put in new ffmpeg codecs, try the command 'grep "_DEV" *.src' in src/ directory
 * 
 * When compiling a static version of ffmpeg libraries, use "lib/run_config_.bash".  Run it in the directory where you have ffmpeg's "configure.bash" script.  It configures ffmpeg with minimal dependencies and with no encoders:
 * 
 * - No dependencies on external libraries
 * - LGPL license
 * - As we are interested only in decoding, and not in transcoding ..
 * - .. all DEcoders are enabled
 * - .. and all ENcoders disabled
 * 
 * Check out also the "lib/config_ffmpeg.py" helper script
 * 
 */


/** @page process_chart Library architecture
 * 
 * H264 decoding is done on the CPU, using the FFmpeg library.  The final operation of interpolation from YUV bitmap into a RGB bitmap of correct (window) size is done on the GPU, using the openGL shading language.
 * 
 * This approach is a nice compromise as it takes some advantage of the GPU by offloading the (heavy) interpolation of (large) bitmaps.
 * 
 * One might think that it would be fancier to do the H264 decoding on the GPU as well, but this is a road to hell - forget about it.
 * 
 * Nv$dia for example, offers H264/5 decoding directly on their GPUs, but then you are dependent on their proprietary implementation, which means that:
 * 
 * - You'll never know how many H264 streams the proprietary graphics drivers is allowed to decode simultaneously.  Such restraints are completely artificial and they are implemented so that you would buy a more expensive "specialized" card.
 * - You'll never know what H264 "profiles" the proprietary driver supports.
 * - There is no way you can even find out these things - no document exists that would reveal which chipset/card supports how many simultaneous H264 streams and which H264 profiles.
 * 
 * So, if you're decoding only one H264 stream, then it might be ok to use proprietary H264 decoding on the GPU (but on the other hand, what's the point if it's only one stream..).  If you want to do some serious parallel streaming (like here), invest on CPUs instead.  
 * 
 * Other possibilities to transfer the H264 decoding completely to the GPU are (not implemented in Valkka at the moment):
 * 
 * - Use Nvidia VDPAU (the API is open source) of the Mesa stack.  In this case you must use X.org drivers for your GPU.  
 * - Create a H264 decoder based on the OpenGL shading language.  This would be cool (and demanding) project.
 * 
 * In Valkka, concurrency in decoding and presenting various streams simultaneously is achieved using multithreading and mutex-protected fifos.  This works roughly as follows:
 * 
 \verbatim
 Live555 thread (LiveThread)     FrameFifo       Decoding threads      OpenGLFrameFifo          OpenGLThread
                                                      
                                                                                               +-------------+ 
                                                                                               |             |
  +---------------------------+                                                                |interpolation|
  | rtsp negotiation          | -> [FIFO] ->      [AVThread] ->                                |timing       |
  | frame composition         | -> [FIFO] ->      [AVthread] ->          [Global FIFO] ->      |presentation |
  |                           | -> [FIFO] ->      [AVthread] ->                                |             |
  +---------------------------+                                                                |             |
                                                                                               +-------------+
 \endverbatim
 * 
 * A general purpose "mother" class Thread has been implemented (see \ref threading_tag) for multithreading schemes and is inherited by:
 * 
 * - LiveThread, for connecting to media sources using the Live555 streaming library, see \ref livethread_tag
 * - AVThread, for decoding streams using the FFMpeg library and uploading them to GPU, see \ref decoding_tag
 * - OpenGLThread, that handles direct memory access to GPU and presents the Frames, based on their timestamps, see \ref openglthread_tag
 * 
 * To get a rough idea how Live555 works, please see \ref live555_page and \ref live_tag.  The livethread produces frames (class Frame), that are passed to mutex-protected fifos (see \ref queues_tag).
 * 
 * Between the threads, frames are passed through series of "filters" (see \ref filters_tag).  Filters can be used to modify the media packets (say, their timestamps for example) and to produce copying and redirection of the stream.  Valkka library filters should not be confused with Live555 sink/source/filters nor with FFmpeg filters - which are completely different things.
 * 
 * For visualization of the media stream plumbings / graphs, we adopt the following notation which you should always use when commenting your python or cpp code:
 * 
 \verbatim
 () == Thread
 {} == FrameFilter
 [] == FrameFifo queue
 \endverbatim
 * 
 * To be more informative, we use:
 * 
 \verbatim
 (N. Thread class name: variable name)
 {N. FrameFilter class name: variable name)
 [N. FrameFifo class name: variable name]
 \endverbatim
 * 
 * A typical thread / framefilter graph would then look like this:
 * 
 \verbatim
 (1.LiveThread:livethread) --> {2.TimestampFrameFilter:myfilter} --> {3.FifoFrameFilter:fifofilter} --> [4.FrameFifo:framefifo] -->> (5.AVThread:avthread) --> ...
 \endverbatim
 * Which means that ..
 * - (1) LiveThread reads the rtsp camera source, passes the frames to filter (2) that corrects the timestamp of the frame.
 * - (2) passes the frames to a special filter (FifoFrameFilter) which feeds a fifo queue (4).
 * - (4) FrameFifo is a class that handles a mutex-proteced fifo and a stack for frames
 * 
 * The whole filter chain from (1) to (4) is simply a callback cascade.  Because of this, the execution of LiveThread (1) is blocked, until the callback chain has been completed.  The callback chain ends to the "thread border", marked with "-->>".  On the "other side" of the thread border, another thread is running independently.
 * 
 * Also, keep in mind the following rule:
 * 
 * - Processes read from mutex-protected fifos (base class for fifos is FrameFifo)
 * - Processes write into filters (base class FrameFilter)
 * 
 * In practice, Thread classes manage their own internal FrameFifo and FifoFrameFilter instances, and things become simpler:
 \verbatim
 * (1.LiveThread:livethread) --> {2.TimestampFrameFilter:myfilter} -->> (3.AVThread:avthread) --> ...
 \endverbatim
 *
 * An input framefilter can be requested with AVThread::getFrameFifo()
 *
 * LiveThread, AVThread and OpenGLThread constructors take a parameter that defines the stack/fifo combination (FrameFifoContext, OpenGLFrameFifoContext).
 *
 * In the case of LiveThread, the API user passes a separate FrameFilter per each requested stream to LiveThread.  That FrameFilter then serves as a starting point for the filter chain.  The last filter in the chain is typically FifoFrameFilter, e.g. a filter that feeds the (modified/filtered) decoded frame to a fifo that is then being consumed by AVThread.
 * 
 * For more details, refer to examples, doxygen documentation and the source code itself.
 * 
 * Remember that example we sketched in the github readme page?  Using our notation, it would look like this:
 * 
 \verbatim
  
  (1.LiveThread:livethread) --> {2.TimestampFrameFilter:myfilter} 
                                   |
                                   +--> {3.ForkFrameFilter:forkfilter}  
                                          |    |
                                          |    |
        through filters, to filesystem <--+    +--->> (6.AVThread:avthread) ---------------------+
                                                                                                 |
                                                                                                 +--> {7.ForkFrameFilter:forkfilter2}  
                                                                                                                  |    |                                                                                    
                                                                                                                  |    |
                                   (10.OpenGLThread:openglthread) <<----------------------------------------------+    +--> .. finally, to analyzing process
                              feeds the video into various X-windoses
  
 \endverbatim
 * - For the various FrameFilters, see \ref filters_tag
 * - For threads, see \ref threading_tag
 *
 * Some more miscellaneous details about the architecture:
 * 
 * - AVThread decoding threads both decode (with FFmpeg) and runs the uploading of the YUV bitmaps to the GPU.  Uploading pixel buffer objects takes place in OpenGLFrameFifo.
 * 
 * - The OpenGLThread thread performs the final interpolation from YUV into RGB bitmap using the openGL shading language
 * 
 * - Fifos are a combination of a fifo queue and a stack: each frame inserted into the fifo is taken from an internal reservoir stack.  If no frames are left in the stack, it means overflow and the fifo/stack is resetted to its initial state
 * 
 * - Reserving items for the fifo *beforehand* and placing them into a reservoir stack avoids constant memory (de)allocations that can become a bottleneck in multithreading schemes.
 * 
 * - This way we also get "graceful overflow" behaviour: in the case the decoding threads or the OpenGL thread being too slow (i.e. if you have too many/heavy streams), the pipeline overflows in a controlled way.
 * 
 * - For a complete walk-through from stream source to x window, check out \ref pipeline
 */



/** @page live555_page Live555 primer
 * 
 * Valkka uses the <a href="http://live555.com">live555 streaming media library</a>.  This page attempts to make it easier for the beginner to get a grasp on live555 (and Valkka).
 * 
 * Live555 is based on event loops and callbacks (event loop and callback "paradigms" to put it in fancy terms).  Events are registered to an event loop.  For example, an event can send a message through TCP socket and set a callback that is executed once a reply has been obtained.  The callbacks then register new events with new callbacks to the event loop and so forth.
 * 
 * An event can be the arrival of a streaming UDP packet.  This triggers a callback that aggregates the packet content to a slice of H264 video, for example.  After this, a new event, that waits for the next packet, is registered to the event loop.
 *
 * Connecting to an ip camera and starting to stream media, would look (very) roughly like this:
 * 
 * -----------------------------------------
 * [ Event loop "tick" number : execution ]
 * 
 * 001 :
 * 
 * 002 : Send RTSP DESCRIBE command through a TCP socket to the ip camera.  Define a callback function ("continueAfterDESCRIBE") that is called once the response is obtained.  Wait for TCP socket using select.
 * 
 * 003 :
 * 
 * 004 : Got response to RTSP DESCRIBE.  Use the callback function ("continueAfterDESCRIBE").  It sends the RTSP SETUP message to the ip camera and defines a callback function that is called once we get response to RTSP SETUP ("continueAfterSETUP").  Wait for TCP socket using select.
 * 
 * 005 : Got response to RTSP SETUP. Use the callback function .. etc.
 * 
 * [Go through the whole RTSP negotiation this way]
 * 
 * ...
 * 
 * 040 : Got a packet of H264 slice from an UDP socket.  Aggregate it to the current H264 slice under construction.  Register an event waiting for the next UDP packet.  Wait for the UDP socket using select.
 * 
 * 042 : Got a packet of H264 slice from an UDP socket.  Aggregate it to the current H264 slice under construction.  Register an event waiting for the next UDP packet.  Wait for the UDP socket using select.
 * 
 * ...
 * 
 * 050 : Got the last packet of a H264 slice from an UDP socket.  Aggregate it to the current H264 slice under construction.  Call afterGettingFrame (of the subclassed MediaSink object) with the H264 slice.  Start a new H264 slice.  Register an event waiting for the next UDP packet.  Wait for the UDP socket using select.
 * 
 * ...
 * 
 * -----------------------------------------
 * 
 * The example above is a rough approximation to what live555 does.  It also considers only one stream coming from a single ip camera.  In practice, live555 "multiplexes" several streams coming from several ip cameras simultaneously and this way achieves concurrency - no threads required !
 * 
 * Libraries using live555 typically "hook up" to the afterGettingFrame callback, by implementing their own MediaSink that receives the composed frame.  The complete frame is then passed on for further processing, decoding and visualization.
 * 
 * Valkka "isolates" live555 library into a separately running thread.  Inside this thread, live555 runs happily and does its magic.  In the thread, we also register a a periodic callback into the live555 event loop (periodic = a callback that re-registers itself periodically).  This callback checks every second, from within the live555 event loop, any incoming commands to the thread via a mutex/condition variable protected message variable.  By sending messages, we can instantiate, shut-down, etc. new RTSP connections *inside* the live555 event loop.  This approach is completely thread-safe.
 * 
 * In Valkka, once the composed frame is obtained, it is passed through series of "frame filters" in a callback cascade.  Each filter does something to the frame, say, corrects the presentation timestamp in some way or does further composition, etc. and then passes the frame to the next filter.  The frame filter callback chain typically ends into a frame filter that inserts the frame into a mutex-protected fifo queue for inter-thread communication (see \ref process_chart).
 * 
 * It is important to remember that while such callback chains are executed, the live555 event loop is paralyzed (see step "050" of the event loop above) !  The callback cascades end typically on a thread or process "border".
 * 
 */


/** @page live_streaming_page Live streaming
 * 
 * Some notes on receiving/sending live streams
 * 
 * 
 * Receiving streams
 * -----------------
 * 
 *- LiveThread::slots_ is a vector of Connection instances.  
 * 
 *- Let's assume there is an RTSPConnection instance in a slot.  The RTSPConnection::client features a ValkkaRTSPClient instance, that is derived from the live555 RTSPClient class.  ValkkaRTSPClient encapsulates all the usual live555 stuff, i.e. response handlers, callbacks, etc.  
 *
 *- ValkkaRTSPClient has the target FrameFilter where the frames are being fed.  ValkkaRTSPClient::livestatus is a reference to a shared variable that is used by LiveThread::periodicTask to see if the client is alive or if it's been destroyed (calling the destructor happens within live555 callback chains) .. this part is a bit cumbersome, so fresh ideas are welcome.
 * 
 *- Signaling with the "outside world" is done by a periodic live555 task LiveThread::periodicTask that reads the signals sent from outside the thread (see LiveThread::handleSignals) 
 * 
 * 
 * Sending streams
 * ---------------
 * 
 * From the point of view of API users, sending frames happens simply by connecting a source to LiveFifo (see the examples).  Behind the scene, frames are sent over the net, by calling LiveFifo::writeCopy - i.e. using the unified approach in Valkka library to handle frames; they are handled in the same way, whether they are passed to the decoder or sent to the screen by using OpenGLFrameFifo (see \ref process_chart).
 * 
 * Sending frames happens roughly as follows:
 *
 *- LiveFifo has a reference to the associated LiveThread
 * 
 *- Let's remember that calls to LiveFifo::writeCopy are done outside the running LiveThread that's sending frames.  Typically by FifoFrameFilter.
 *
 *- LiveFifo::writeCopy calls LiveThread::triggerGotFrames which triggers an event using the scheduler and the triggerEvent method (this is the only allowed way to launch events outside the live555 event loop)
 *
 *- The triggered event corresponds to LiveThread::gotFramesEvent
 *
 *- LiveThread::gotFramesEvent inserts a new (immediate) task to the live555 event loop, namely LiveThread::readFrameFifoTask
 *
 *- LiveThread::readFrameFifoTask (i) calls LiveThread::handleFrame and (ii) re-schedules itself if there are more than one frame in the fifo
 *
 *- LiveThread::handleFrame checks the slot of the outgoing frame, takes the corresponding Outbound instance and calls Outbound::handleFrame
 *
 * SDP Streams
 * -----------
 * 
 * Streams sent directly to UDP ports, as defined in an SDP file ("SDP" streams)
 * 
 *- SDPOutbound has a set of Stream instances in Outbound::streams.  There is a Stream instance per media substream.
 *
 *- Stream instances encapsulate the usual live555 stuff per substream: RTPSink, RTCPInstance, Groupsock, FramedSource, etc.
 *
 *- Each Stream has also a reference to LiveFifo and Stream::buffer_source which is a BufferSource instance.
 *
 *- BufferSource is a subclass of live555 FramedSource that is used for sending frames. 
 *
 *- BufferSource has it's own internal fifo BufferSource::internal_fifo.  This fifo is used by BufferSource::doGetNextFrame, as it should in live555, i.e. if there are no frames left in the fifo, then BufferSource::doGetNextFrame exits immediately, otherwise it calls FramedSource::afterGetting and if necessary, re-schedules itself
 *  
 *- To summarize, the call chain looks like this: LiveThread::handleFrame => Outbound::handleFrame => Stream::handleFrame (transfers the frame from LiveFifo to BufferSource::internal_fifo) => BufferSource::handleFrame => FramedSource::afterGetting
 *
 *- The Live555 filterchain looks like this:
 \verbatim 
 buffer_source  =new BufferSource(env, fifo, 0, 0, 4); // nalstamp offset: 4
 terminal       =H264VideoStreamDiscreteFramer::createNew(env, buffer_source);
 
 sink           = H264VideoRTPSink::createNew(env,rtpGroupsock, 96);

 sink->startPlaying(*(terminal), this->afterPlaying, this);
 \endverbatim
 *
 *- Frames flow like this: 
 *BufferSource => H264VideoStreamDiscreteFramer => H264VideoRTPSink
 * 
 *
 * This diagram, where "{}" means enclosing object will help you to understand this:
 \verbatim
 BufferSource (.., fifo) {
    - Live555 FramedSource class with method "doGetNextFrame"
    - Recycles frames back to fifo (in doGetNextFrame)
 }
 
 Stream {
    RTPSink,
    RTCPInstance,
    Groupsock,
    BufferSource *buffer_source,
    FrameFifo &fifo,
    
    methods:
        startPlaying
            - issues startPlaying on the live555 
              event loop (for the internal live555 
              filterchain defined in the Stream subclasses)
        afterPlaying
            - a live555 callback
 }
 
 H264 : public Stream {
    - Instantiates buffer_source = new BufferSource (.., fifo)
    - Creates the live555 internal filterchain
 }
 \endverbatim
 *
 * RTSP Server
 * -----------
 *
 * Some misc. observations / code walkthrough:
 *
 * - Inheritance:  H264ServerMediaSubsession => ValkkaServerMediaSubsession => OnDemandServerMediaSubsession 
 * - H264ServerMediaSubsession is placed into LiveThread::server (which is RTSPServer instance)
 * - H264ServerMediaSubsession::createNewStreamSource is called from OnDemandServerMediaSubsession::sdpLines when a DESCRIBE request is done on the RTSPServer
 * - OnDemandServerMediaSubsession::closeStreamSource calls Medium::close on inputsource it obtained from H264ServerMediaSubsession::createNewStreamSource, when TEARDOWN is issued on the RTSPServer.
 * - .. this basically closes our BufferSource instance (that's using a fifo), so we need to check if BufferSource has been destructed
 * - .. this is done using a reference variable that's given to BufferSource (ValkkaServerMediaSubsession::source_alive).  That variable is checked before calling BufferSource::handleFrame.
 * - There is a tricky inner event loop that generates sps/pps info into the sdp string at H264ServerMediaSubsession::getAuxSDPLine .. (as this has been hacked from H264VideoFileServerMediaSubsession).  That can get stuck sometimes (!)
 * - Media sessions can be removed from the server with RTSPServer::removeServerMediaSession(media_session)
 *
 * To get this into one's head, let's take a look at this diagram.  "{}" means enclosing object:
 \verbatim
 RTSPServer {
 
    H264ServerMediaSubsession {
        - member "fifo" is a reference to FrameFifo
    
        - method : createNewStreamSource
            - creates buffer_source = BufferSource (.., fifo)
            - creates H264VideoStreamDiscreteFramer(buffer_source)
            => returns to RTPServer: H264VideoStreamDiscreteFramer(buffer_source)
            
        - inherited method sdpLines
        - inherited method closeStreamSource
    }
 }
 \endverbatim
 */


/********* GROUP/MODULE DEFINITIONS ***************/


/** @defgroup frames_tag the frame class
 * 
 * Things related to Frame
 *
 */


/** @defgroup filters_tag available framefilters
 * 
 * Things related to FrameFilters
 *
 */


/** @defgroup live_tag live555 bridge
 * 
 * How we're using the live555 API
 *
 */


/** @defgroup file_tag file streaming
 * 
 * Things related to streaming from/into files
 * 
 * At the moment, reading from files is done by a single thread that multiplexes the reading from various files, FileThread
 * 
 * Writing is done simply by writing to a special FrameFilter, namely, the FileFrameFilter.  In the future, we'd like to do this in a separate thread as well (include writing into FileThread, maybe)
 * 
 */


/** @defgroup threading_tag multithreading
 * 
 * - Threads are running independently at the "backend"
 * - They are started, controlled and stopped from the "frontend Python side"
 * - This image illustrates the situation:
 * 
 \verbatim
 +-------------------------+         +-------------------------+-------------------------+           
 | "backend" thread        |         |                 "frontend" thread                 |
 | c++ thread              |         |                         |    Python side method   |
 | running independently   |         |     c++ side            |    (hold GIL)           |
 |                         |         |                         |     |                   |
 | - do stuff &            |         |                   <-----|-----+                   |
 |   get messages from     | [queue] |   - send message to     |                         |
 |   queue                 |         |     queue         >-----|-----+                   |
 | - perform callbacks     |         |                         |     |                   |
 |   to python side        |         |   - for backend         |   continue in Python    |
 |   (obtain GIL)          |         |     termination         |   side & exit           |
 |                         |         |     wait for thread     |   (release GIL)         |
 |                         |         |     join                |                         |
 |                         |         |                         |                         |
 +-------------------------+         +-------------------------+-------------------------+
 \endverbatim
 * 
 * The frontend's "Python side" API methods are typically tagged with "Call", i.e. they are named "startCall", "stopCall", etc. and are automatically generated from the c++ methods using SWIG.
 *
 * The backend runs std::thread or pthread whose target method is Thread::mainRun().  The Thread class is a prototype for implementing thread-safe multithreading classes.  There are three pure virtual classes, namely:
 * 
 * 1. Thread::preRun() allocates necessary resources for the Thread (if not reserved at constructor time)
 * 2. Thread::run() uses the resources and does an infinite loop.  It listens to requests and executes Thread's methods
 * 3. Thread::postRun() deallocates resources
 * 
 * The method Thread::mainRun() simply does the sequence (1-3)
 *
 * Since we're dealing here with extending python code in c++ (frontend) and with calling python callbacks from c++ (backend), extra care must be taken with the Python Global Interpreter Lock (GIL)
 *
 * Frontend methods hold the Python GIL (as they are just normal python methods), which is kept during the whole execution of the c++ "frontend" part (i.e. no Py_BEGIN_ALLOW_THREADS here).  These are just fast calls that exit once they have sent a signal to the message queue.
 *
 * An exception to this rule is the special method Thread::requestStopCall() which, after sending a termination signal to the backend, waits for the thread join (for joining Thread::mainRun()).
 *
 * If at this moment, the backend is trying to perform a python callback, a deadlock will occur: the frontend is holding the python GIL, while the backend is waiting for its release.
 *
 * All Python callbacks from the backend after Thread::run() has been exited, should then be performed in the Thread's destructor, and without touching the GIL (destructors are typically evoked by the Python garbage collector, i.e. the GIL is being hold from the Python side)
 *
 * For practical Thread implementations, see for example LiveThread and OpenGLThread
 * 
 * Thread can be bound to a particular processor core, if needed.
 * 
 * 
 */


/** @defgroup shmem_tag shmem
 * 
 * Posix shared memory and semaphores.  Using shared memory, multiprocesses, especially python processes, can interchange frames.  Used typically for sending frames from AVThread to a python multiprocess at the python API side.
 * 
 * According to my experience, this is good for passing, say, one full-hd frame per second (good enough to do ok image analysis).  However, don't expect this to work for high-throughput 30 fps per second for several cameras..!
 * 
 * In the SharedMemRingBuffer the name is important.  It identifies the POSIX shared memory segments both in the server and client side.
 * 
 */


/** @defgroup livethread_tag livethread
 * The LiveThread class implements a "producer" thread (see \ref threading_tag) that feeds frames into a FrameFilter (see \ref frames_tag). 
 * The FrameFilter chain feeds finally a FrameFifo, which is being read by a "consumer" thread (typically a AVThread).
 * 
 * After the LiveThread has been started with LiveThread::startCall, connections can be defined by instantiating a LiveConnectionContext and by passing it to LiveThread::registerStream
 * 
 */


/** @defgroup decoding_tag decoding
 * 
 * DecoderBase is the base class for various decoders.  AVThread is a Thread subclass that consumes frames, decodes them using DecoderBase instances and passes them along for the OpenGLThread.
 *
 */


/** @defgroup openglthread_tag opengl
 * 
 * Things related to OpenGL
 * 
 * 
 */


/** @defgroup audio_tag audio
 * 
 * Audio reproduction
 * 
 * Sorry, no audio at the moment..!
 * 
 * The plan for this is as follows: redirect, using filterchains, audio to a separate Thread.  The name of that thread will be ALSAThread, etc.  Analogical to OpenGLThread.
 * 
 */


/** @defgroup queues_tag queues and fifos
 * 
 * Multiprocessing queues/fifos.  The base class is FrameFifo.  Special derived classes (LiveFifo, OpenGLFrameFifo) are usually created and managed by certain derived Thread classes (LiveThread, OpenGLThread).
 *
 */


/** @page pipeline Code walkthrough: rendering
 * 
 * From RTSP stream to X-window.  
 *
 * Let's recall a typical filtergraph from \ref process_chart
 \verbatim
 (1.LiveThread) --> {2.FifoFrameFilter} --> [3.FrameFifo] -->> (4.AVThread) --> {5.FifoFrameFilter} --> [6.OpenGLFrameFifo] -->> (7.OpenGLThread)
 \endverbatim
 *
 * In detail, frames are transported from (1) to (7) like this:
 * 
 * 1 LiveThread
 * - Frame::frametype set to FrameType::h264
 * - Frame::payload has the raw data
 * 
 * 4 AVThread
 * - Uses data in Frame::payload
 * - Sets Frame::frametype to FrameType::avframe
 * 
 * 6 OpenGLFrameFifo
 * - Has internal stacks of YUVPBO objects (with direct memory access to GPU)
 * - A frame from one of the stacks (corresponding to the Frame resolution) is reserved
 * - OpenGLFrameFifo::prepareAVFrame uploads data to GPU
 * 
 * 7 OpenGLThread
 * - Handles all X11 and GLX calls
 * - Reads OpenGLFrameFifo, presents frames according to their timestamps
 * - Once a Frame has been presented, it is recycled back to OpenGLFrameFifo
 *  
 * OpenGLThread uses the following classes:
 * 
 * - SlotContext == Unique data related to a stream.  Textures.
 * - RenderGroup == Corresponds to a unique X window.  Has a list of RenderContext instances.
 * - RenderContext == How to render a texture in OpenGL:  vertex array object (VAO), vertex buffer object (VBO), transformation matrix, etc.  Has a reference to an activated SlotContext.  This is the render target for your RTSP stream!
 * 
 * Internal data structures of OpenGLThread:
 * 
 * OpenGLThread::slots_        == A vector.  A SlotContext for each slot.  Warehouses SlotContext instances
 * 
 * OpenGLThread::render_groups == Mapping. Window_id => RenderGroup mapping.  Warehouses the RenderGroup instances (each RenderGroup instance warehouses RenderContext instances)
 * 
 * OpenGLThread::render_lists  == A vector.  There is one element for each slot.  Each element is a list of references to RenderGroup instances.  Only RenderGroup(s) that have some RenderContext(es) should be in this list.
 * 
 * So, OpenGLThread::run reads a new frame from OpenGLThread::infifo.  It is passed to the presentation queue, i.e. to OpenGLThread::presfifo.  OpenGLThread::handleFifo is eventually called.
 * 
 * OpenGLThread::handleFifo inspects Frame::n_slot => takes a SlotContext from OpenGLThread::slots_.  Preload the textures to SlotContext::yuvtex.
 * So, textures are now loaded and ready to go. Now we have to find all render contexes that use these textures (there might be many of them).
 * 
 * Using Frame::n_slot, pick a list from OpenGLThread::render_lists (there is one list for each slot)
 * 
 * Run through that list: each element is a RenderGroup (remember that each RenderGroup has a list of RenderContexes)
 * 
 * RenderGroup has method RenderGroup::render.  That method fixes the current X window for manipulation.  So, X window has been chosen.  Let's draw into it.
 * 
 * Then RenderGroup runs through all its RenderContex(s).  For each RenderContext, RenderContext::render is called
 *
 * RenderContext has vertex array, vertex buffer objects, transformation matrix, etc., everything needed for rendering a YUV image in any way you wish (straight rectangle, twisted, whatever). 
 * 
 * RenderContext also has a reference to a valid SlotContext: RenderContext::slot_context.  From there we get the Shader and YUVTEX (pre-loaded textures) which are now used together with the vertex array, vertex buffer objects, transformation matrix, etc.
 * 
 * Finally, render the texture with glDrawElements in RenderContext::bindVertexArray.
 * 
 * Recycle frame back to OpenGLFrameFifo
*/


/** @page pipeline2 Code walkthrough: OpenGL
 * 
 * 
 * Call structure for some GLX and OpenGL calls
 * 
 * OpenGLThread::handleFifo 
 * - OpenGLThread::loadTEX 
 *     - loadYUVTEX 
 *         - glBindBufferARB, glBindTexture, glTexSubImage2D 
 * - OpenGLThread::render (SlotNumber n_slot)
 *     - OpenGLThread::render_lists = list of RenderGroup (s) per slot
 *         - RenderGroup::render glXMakeCurrent, XGetWindowAttributes, glViewport, glClear 
 *             - RenderGroup::render_contexes = list of RenderContext (s) 
 *               - RenderContext::render 
 *                 - Shader::use 
 *                 - OpenGLThread::bindTextures() 
 *                   - glActiveTexture, glBindTexture, glUniform1i 
 *                 - OpenGLThread::bindVars() 
 *                   - glUniformMatrix4fv 
 *                 - OpenGLThread::bindVertexArray() 
 *                   - glBindVertexArray, glDrawElements, glBindVertexArray 
 *             - glXSwapBuffers 
 * 
 * SlotContext manages textures: 
 *                                   
 * - SlotContext::yuvtex
 * - SlotContext::statictex
 * 
 * Several RenderContext instances (i.e. mappings) can refer to the same SlotContext.
 * 
 * This way we can map the same texture into several x-windows and also into "picture-in-picture" schemes.  Let's state this idea graphically:
 * 
 *\verbatim 
 * 
 * 
 * 
 *                                        (RenderGroup 1 = x window 1)
 *                                        +------------+
 *                                        |+-----+     |
 *                                        ||     |     |  ("map slot 1 to RenderGroup 1")
 *                  +--- RenderContext---->|     |     |
 *                  |                     |+-----+     |
 * SlotContext -----+                     +------------+
 *                  |                     
 *                  |                     +------------+
 *                  +-- RenderContext---->|            |  ("map slot 1 to RenderGroup 2")
 *                                        |            |
 *                                        |            |
 *                                        |            |
 *                                        +------------+
 *                                        (RenderGroup 2 = x window 2) 
 * 
 *\endverbatim 
 * 
 */


/** @page timing Presention timing and playing
 * 
 * Notation convention for handling presentation timestamps (PTS)
 * 
 * File streams
 * ------------
 * 
 * The underscore "_" tags "stream time". %Stream timestamps are timestamps on each recorded frame: they can be far in the past or in the future.
 * 
 * The following equation holds:
 * 
 * (t_-t_0) = (t-t0)
 * 
 * where:
 * 
 * Symbol         | Explanation
 * -------------- | ------------------------------
 * t_             | the file stream time
 * t              | the wallclock time
 * t0_            | file stream reference time
 * t0             | wallclock reference time
 * 
 * 
 * t0 is measured at the time instant when t0_ is set.  This is done at seek.
 * 
 * We define a "reference time":
 * 
 * reftime = (t0 - t0_)
 * 
 * The we get a frame's wallclock time like this:
 * 
 * t = t_ + reftime
 * 
 * Where t_ is the frame's timestamp in the stream.  You can think it as we're correcting a recorded frame's timestamp to present time.
 * 
 * [ check: t = t_ + reftime <=> t = t_ + t0 - t0_ <=> t - t0 <=> t_ - t0_ ]
 * 
 * 
 * Different actions for file streams
 * 
 * For file streams, a target_time_ is set.  Frames are consumed until t_>=target_time_
 * 
 * Action     | What it does
 * ---------- | -----------------------------------------------
 * open       | Open file, do seek(0_)
 * seek       | set target_time_, once t_>=target_time_, stop
 * play       | set target_time_, keep consuming frames with t_>=target_time_.  Update target_time_.
 * stop       | Stop
 * close      | Deregister file
 * 
 * 
 * 
 * Realtime streams 
 * ----------------
 * 
 * Symbol         | Explanation
 * -------------- | ------------------------------
 * t_             | frame's timestamp
 * t              | the wallclock time
 * tb             | buffering time
 * 
 * Define "relative timestamp" :
 * 
 * trel = t_ - t + tb = t_ - (t-tb) = t_ - delta
 * 
 * 
 *\verbatim
 *             [fifo]
 * => in                       => out
 * [young                        old]
 * 
 * absolute timestamps
 * 90  80  70  60  50  40  30  20  10
 *
 * relative timestamps trel:
 * .. with delta=(t-tb)=45
 *                   |
 * 45  35  25  15  05 -15 -25 -35 -35
 * 
 * negative values == older frames, positive values == younger frames
 *\endverbatim
 * 
 * - negative frames are late: "presentation edge" is at 0
 * - increasing buffering time moves the "presentation edge" to the right == less late frames
 * - remember: large buftime will flood the fifo and you'll run out of frames in the stacks
 * - 50-100 milliseconds is a nice buftime value
 * 
 * extreme cases:
 * 
 * - all negative == all frames too old   (i.e. due)               
 * - all positive == all frames too young (i.e. in the future)
 *
 */
 

/** @page valkkafs ValkkaFS
 * 
 * Notes on the block and timestamp scheme
 * 
 * - Stream number 0 is always the "leading codec"
 *
\verbatim

    A, a = frames from source A (mayor letter = KEY frame.  Designates time start, i.e. sps packet, etc.)
    B, b = frames from source B
    etc.
    
    Assume that if cameras are on-line, there is a key-frame from each camera in every block
    

                                                                BlockTable
                                                                
                                                                k-max, maximum KEY frame timestamp in block
                                                                max, maximum frame timestamp in block
                                                                
    time ->
                                                                k-max max       block
                                                                
    01 02 03 04 05 06 07 08 09 10 11 12 13 14                   12    14        10
    b  b  d  D  a  A  a  c  c  a  C  B  b  b                    

    15 16 17 18 19 20 21 22 23 24 25 26 27 28                   26    28        11
    a  a  B  c  A  C  b  b  c  b  a  D  d  a 

    29 30 31 32 33 34 35 36 37 38 39 40 41 42                   37    42        12
    A  b  B  c  d  C  d  d  D  a  a  b  c  a

\endverbatim
 *    
 * Request time (21, 29) == (seek_time, end_time)
 *
 * ==>
 *
 * Lower limit: we need that all cameras have stamped a key frame with =< 21.  This way we can seek to t = 21.
 * 
 * { all keyframes <= seek_time, at least if max(all keyframes) <= seek_time }
 *
 * if any of the cameras has stamped > 21, needs earlier block 
 * 
 * --> start from block 10 (search: last block of all blocks that have k-max =< 21)
 *  
 * Higher limit: all cameras have stamped >= 29 for any frame
 * 
 * if any of the cameras has stamped < 29, needs later block
 * 
 * --> last block to include is 12 (search: first block of all blocks that have max >= 29)
 * 
 */



/** @page filesystem ValkkaFSManager
 * 
 * Writing, reading and caching frames
 * 
 * The level 2 API Python class ValkkaFSManager, uses several level 1 API (core) Python class objects
 * 
 * - core.ValkkaFS : blocktable and book-keeping
 * - core.ValkkaFSReaderThread : reads frames from the file or block device
 * - core.ValkkaFSCacherThread : caches the read frames into memory (typically several blocks of frames)
 * - core.ValkkaFSWriterThread : writes frames into the file or block device
 * 
 * - Frames are requested on per-block basis from core.ValkkaFSReaderThread.  It feeds frames to core.ValkkaFSCacherThread
 * - Seek, play and stop operations take place within the cached frames in core.ValkkaFSCacherThread
 * - All Threads share a common core.ValkkaFS object that has the blocktable and is also visible at the Python side
 * 
 * The logic of requesting certain blocks in order to show (and buffer) frames for a certain time instant is handled completely at the python side
 * 
 * This orchestration is handled by the level 2 API Python class ValkkaFSManager.
 *
 * Let's use the following pseudocode notation, to see how objects are contained within other objects:
\verbatim
classname(init parameter) {
    classnames of contained objects
}
\endverbatim
 *
 * This is how it looks like.  Let's hope you'll get the big picture.  :)
 *
\verbatim

- cpp threads run & originate python callbacks
- core.ValkkaFSWriterThread "drives" core.ValkkaFS which emits callbacks

api2.ValkkaFSManager(api2.ValkkaFS) {
    
    1: api2.ValkkaFS {
        core.ValkkaFS
        
        # c++ => python callbacks
        def new_block_cb__(propagate, par):
            - launched from cpp core.ValkkaFS.writeBlock (pycall) when a block is finished
            - propagate indicates if further callbacks should be evoked
            - par is an integer (block number) or an error string
            - calls self.block_cb for callback propagation

        # python => c++ calls
        def getBlockTable():
            - updates python side blocktable (numpy array)
            - calls cpp-side core.ValkkaFS.setArrayCall(self.blocktable_)
            - returns self.blocktable_
        }
            
    2: core.ValkkaFSReaderThread {
        core.ValkkaFS
        - writes to framefilter that is got from core.FileCacherThread.getFrameFilter() [4]
        - frames are requested on per-block basis

        # python => c++ calls
        pullBlocksPyCall(block_list) => [signal to thread]
            => pullBlocks
                - Writes frames to its outgoing framefilter 
                 (typically connected to FileCacherThread::getFrameFilter())
                - results in launching core.FileCacherThread.switchCache => pyfunc2
                  => timeLimitsCallback__
        }

    3: core.ValkkaFSWriterThread {
        core.ValkkaFS
        - input framefilter can be requested with getFrameFilter()
        }

    4: core.FileCacherThread {
        - gets frames from core.ValkkaFSReaderThread via input framefilter
        - caches frames
        - receives seek, play, stop, operations
        - send batches of frames downstream (to output filter)
        }   
    
    # c++ => python callbacks
    def timeCallback__(mstime: int):
        - originates from core.FileCacherThread.run (pyfunc)
        - once per 300 ms
        - calls:
            => self.readBlockTableIf()
                => if necessary, calls self.readBlockTable()
                    => self.blocktable = api2.ValkkaFS.getBlockTable() [1]
                        => cpp core.ValkkaFS.setArrayCall(self.blocktable_)

            => self.reqBlocks(mstimestamp)
                => core.ValkkaFSReaderThread.pullBlocksPyCall(block_list)

    def timeLimitsCallback__(tup: tuple):
        - originates from core.FileCacherThread.switchCache (pyfunc2)
        - sent when frame cache has been updated
    
    
    # some important methods:

    def setBlockCallback(cb):
        define how api2.ValkkaFS.new_block_cb__ is continued
        (by default, no callback chain)
    
    def setOutput(_id, slot, framefilter [**]):
        """Set id => slot mapping.  Set output framefilter
        """
        core.ValkkaFSReaderThread.setSlotIdCall(slot, _id)  # ID-TO-SLOT MAPPING
        ctx = core.FileStreamContext(slot, framefilter)     # SLOT-TO-FRAMEFILTER MAPPING [**]
        core.FileCacherThread.registerStreamCall(ctx)

    def setInput(_id, slot):
        core.ValkkaFSWriterThread.setSlotIdCall(_id, slot)

    def getInputFrameFilter():
        return ValkkaFSWriterThread.getFrameFilter()
    
    }

\endverbatim
 * 
 * Frames are transported like this:
 *
\verbatim
outgoing frames:
    
    core.ValkkaFSReaderThread [2] --> core.FileCacherThread [4] --> output framefilter [**]
    
     - Request blocks of frames        - Set seek point, play,
       to be sent downstream             stop, etc. 
     - Uses shared core.ValkkaFS      
       instance
    
    
incoming frames:

    --> core.ValkkaFSWriterThread.getFrameFilter() --> core.ValkkaFSWriterThread
                                                       
                                                       - Updates shared core.ValkkaFS
                                                         instance
\endverbatim
 *
 *
 */
 
/** @page filesystem cpp / Python callbacks
 * 
 * ValkkaFS is using both cpp-to-python and python-to-cpp calls
 * 
 * This can get tricky, and care must be taken to avoid nasty deadlocks due to the Python Global Interpreter Lock (GIL) 
 * 
 * The cpp part of the code can decide to call some python code, that has been defined in the main thread Python part.  This is done by the cpp code "autonomously" and is not initiated from the python side. (*)
 * 
 * Python part might evoke some cpp code. (**)
 * 
 * Special care must be taken with "callback cascades" that start from cpp and end up back to the cpp side again.
 * 
 * The acquisition and release of GIL is illustrated in the following graph.
\verbatim

(*)      = callback from cpp to Python
(**)     = callback from Python to cpp
|        = the instance holding the GIL
DEADLOCK = example deadlock situations

A program using both (*) and (**) callbacks:


            main thread (Python)           cpp thread
            
               |
               |
               |
                                           | acquire GIL (*) 
                                           | - call python method, defined at main thread
                                           | release GIL
               |
               |
               |
                                           | acquire GIL (*)
                                           | - call python method, defined at main thread
                                           |   - that python method might call cpp code which tries to acquire GIL => DEADLOCK!
                                           | release GIL
            
               |
               | 
               | call swig-wrapped   
               | cpp method (**)           do not acquire GIL           
               |                           as it's being hold by the main thread
               |                           return
               |
                                           | acquire GIL (*)
                                           | - call python method, defined at main thread
                                           | release GIL 
               
\endverbatim




\verbatim
            
            
            
            main thread (Python)           ValkkaFSWriterThread    
            
              |
              |
                                           | acquire GIL
                                           | - call ValkkaFSWriterThread::pyfunc
                                           |   - In the python side, this is set to valkka.api2.valkkafs.new_block_cb__
                                           |   - .. which in turn continues the callback-cascade into ValkkaFS::setArrayCall
                                           | release GIL
              |
              |
            
              | requestStopCall           requestStopCall (cpp side)
              | (python side)             - sends message to thread's message queue
              |                           - returns immediately
             
              |                           exits thread's running loop
               
              | waitStopCall              preJoin 
              |                           => saveCurrentBlock 
              |                              => valkkaFS::writeBlock 
              |                                 - should not acquire GIL as this has been 
              |                                   requested from the python side
              |                           thread join & exit
            
            --------------------------------------------------
            
                                           | CacheStream::run
                                           | pyfunc
                                                => valkka.api2.ValkkaFSManager.timeCallback__(mstime)
                                                    => valkka.api2.ValkkaFSManager.stop()
                                                        => ValkkaFSWriterThread::stopStreamsCall() # this makes any sense?
                                                - it's ok to call other thread's functions, as long as they don't return the callback chain to python
                                                
            
            
            
\endverbatim
 *
 */


/** @page Caching Frames
 * 
 * 
 * FileCacherThread::run
 * 
 * 
 * 
\verbatim


walltime == wallclock time 

target_mstimestamp_ = 0 
    current target frametime, transformed from current walltime : stopStreams, seekStreams, run

next = NULL 
    points to next frame : run
    
reftime = 0 
    value for transforming frametime <-> walltime : playStreams, seekStreams, run

next_mstimestamp = 0 : run
    timestamp of the next frame in wallclock time

    
while True:
    
    if next != NULL and reftime > 0:
        # next frame exists and we can get it's wallclock time
        next_mstimestamp = next.mstimestamp + reftime # from frametime to walltime
        # => calculate timeout to the next frame
        timeout = ..
    else:
        timeout = default_timeout
    
    f = infifo.read(timeout)
    
    # there's either a frame or this was a timeout
    
    if TIMEOUT:
        pass
        
    elif GOT_FRAME:
        
        if f is signal:
            # handle the signal => seekStreams, playStreams, stopStreams
            
        elif f is marker:
            if marker == TM_START # transmission start
                # do nothing .. frames are flowing to tmp cache
                
            elif marker == TM_END # transmission end
                switchCache() # tmp cache becomes the play_cache
                next = NULL
                if target_mstimestamp_ <= 0: # no seek time set yet
                    # this can happen: seek has not yet called, but block transmission has been requested
                    # that's fixed with the next seek call
                    print WARNING
                else:
                    if state == SEEK or state == STOP:
                        i = play_cache->keySeek
                    elif state == PLAY:
                        i = play_cache->seek
                    else:
                        print WARNING
                        
                    if seek succeeded:
                        next = play_cache.pullNextFrame()
                        
                    if next != None:
                        walltime = getCurrentMsTimestamp();
                        reftime = walltime - target_mstimestamp_; # match walltime to target frametime
                        
        
    # frame handled
    
    mstime = getCurrentMsTimestamp() # update current time
    
    if reftime > 0 and & state == PLAY: # walltime get's updated constantly
        walltime = mstime
        target_mstimestamp_ = walltime - reftime; # update current frametime
            
    # FRAME PULL LOOP: pull all necessary frames from the cache
    while next != NULL and (next_mstimestamp - walltime) <= 0:
        
        if reftime == 0: # no reftime set .. take it from the first frame
            reftime = mstime - target_mstimestamp_
            stopStreams() # state => STOP, send SetupFrames
        
        # create & send a SetupFrame for decoder init if necessary
        ...
        
        # send frame to correct framefilter with this timestamp:
        frame.mstimestamp = next.mstimestamp + reftime # from frametime to walltime
        
        next = play_cache.pullNextFrame()
        if next != NULL:
            next_mstimestamp = next.mstimestamp + reftime # from frametime to walltime
            
        if next == NULL or next_mstimestamp - walltime > 0: # while loop is about to break
            if state == SEEK
                stopStreams() # state => STOP, send SetupFrames
    
    # handle signals etc


- seek sets the reftime
- play : stream's been stopped, so there's a fixed target_mstimestamp_ => get reftime == walltime <-> frametime mapping from there
- stop : stream's been playing / seeking, so there's reftime (wall-time <-> frametime mapping) => get target_mstimestamp_ from walltime

stopStreams:
    if reftime > 0:
        walltime = getCurrentMsTimestamp()
        target_mstimestamp_ = walltime - reftime # get current frametime from walltime
        
playStreams:
    if target_mstimestamp_ > 0:
        walltime = getCurrentMsTimestamp()
        reftime = walltime - target_mstimestamp_ # get reftime from current frametime
        
seekStreams(mstimestamp_): # if target frame time mstimestamp_ is found in play_cache, do immediate seek, otherwise, set reftime = 0
    target_mstimestamp_ = mstimestamp_
    if play_cache has the target frame, then:
        walltime = getCurrentMsTimestamp()
        reftime = walltime - target_mstimestamp_
    else:
        reftime = 0
    




\endverbatim
 * 
 * 
 * 
 */
 
 
/** @page Sharing frames from python
 
\verbatim

valkka.api2.shmem.ShmemRGBServer.


\endverbatim
*/



