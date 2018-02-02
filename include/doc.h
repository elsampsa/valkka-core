/*
 * doc.h : Dummy header file for doxygen documentation
 * 
 * Copyright 2017, 2018 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>
 *
 */

/** 
 *  @file    doc.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.0 
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
 * Valkka - Massive video streaming for linux
 * ------------------------------------------
 *
 * The long-term goal of this project is to provide open source video surveillance, management and analysis (VSMA) systems in Linux environment, and to be able to create them with the combination of Python3 and Qt (i.e. PyQt).  The library has both Cpp and Python3 API (the latter is recommended).
 *
 * Warning: If you simply want to use Valkka's python3 API, you're in the wrong place.  This doxygen generated documentation is for people who wan't to develop Valkka core library.  
 * 
 * For using Valkka API, just download the examples git repository and start coding in python.  Said that, however, you should read about \ref process_chart
 * 
 * 
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
 * Copyright 2017, 2018 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
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
        through filters, to filesystem <--+    +--> {4.FifoFrameFilter:fifofilter} --> [5.FrameFifo:framefifo] -->> (6.AVThread:avthread)
                                                                                                                              |
                                                                                                                              |
                                                                                                                              +--> {7.ForkFrameFilter:forkfilter2}  
                                                                                                                                            |    |                                                                                    
                                                                                                                                            |    |
                                   (10.OpenGLThread:openglthread) <<-- [9.OpenGLFrameFifo:gl_fifo] <-- {8.FifoFrameFilter:gl_in_filter}  <--+    +--> .. finally, to analyzing process
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


/********* GROUP/MODULE DEFINITIONS ***************/


/** @defgroup frames_tag
 * 
 * Things related to Frame
 *
 */


/** @defgroup filters_tag
 * 
 * Things related to FrameFilters
 *
 */


/** @defgroup live_tag
 * 
 * Things related to Live555
 *
 */

/** @defgroup threading_tag
 * 
 * The Thread class is a prototype for implementing thread-safe multithreading classes.  There are three pure virtual classes, namely:
 * 
 * 1. Thread::preRun() allocates necessary resources for the Thread (of course some resources may be reserved at constructor time)
 * 2. Thread::run() uses the resources and does an infinite loop.  It listens to requests and executes Thread's methods
 * 3. Thread::postRun() deallocates resources
 * 
 * The method Thread::mainRun() simply does the sequence (1-3)
 * 
 * Controlling the thread, once it has been started, follows these principles:
 * 
 * 1. In the "front-end", an API method is called with a some context instance "ctx" (say, ip address of the camera, etc.)
 * 2. That API method, in turn, encapsulates the context instance into Thread::SignalContext (that is characteristic of each thread)
 * 3. .. and uses the thread-safe method Thread::sendSignal(signal_context), to place that request into the request queue Thread::signal_fifo
 * 4. At the "back-end" (i.e. "inside" the running thread), Thread::run() is doing it's infinite loop and get's a new request from Thread::signal_fifo
 * 5. Thread::run() uses Thread::handleSignal that delegates the request to one of Thread's methods
 * 
 * To make distinction between "front-end" (API) and "back-end" (internal) methods, the API method names should be designated with the ending "Call".  
 * 
 * The two API methods in the prototype class, Thread::startCall and Thread::stopCall start and stop the thread, respectively.
 * 
 * When developing and debugging Thread classes, one can write test programs like this: 
 * 
 * Thread::preRun(); <br>
 * [call here thread's internal methods] <br>
 * Thread::postRun(); <br>
 * 
 * For practical Thread implementations, see for examṕle LiveThread and OpenGLThread
 * 
 */

/** @defgroup livethread_tag
 * The LiveThread class implements a "producer" thread (see \ref threading_tag) that feeds frames into a FrameFilter (see \ref frames_tag). 
 * The FrameFilter chain feeds finally a FrameFifo, which is being read by a "consumer" thread (typically a AVThread).
 * 
 * After the LiveThread has been started with LiveThread::startCall, connections can be defined by instantiating a LiveConnectionContext and by passing it to LiveThread::registerStream
 * 
 */

/** @defgroup decoding_tag
 * 
 * DecoderBase is the base class for various decoders.  AVThread is a Thread subclass that consumes frames, decodes them using DecoderBase instances and passes them along for the OpenGLThread.
 *
 */

/** @defgroup openglthread_tag
 * 
 * Things related to OpenGLThread: GLX, OpenGL, etc.
 * 
 * 
 */

/** @defgroup queues_tag
 * 
 * Multiprocessing queues/fifos
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
 * - OpenGLThread::render 
 *     - OpenGLThread::render_lists = list of RenderGroup (s) 
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
 */


/** @page timing Presention timing and playing
 * 
 * Notation convention for handling presentation timestamps (PTS)
 * 
 * File streams
 * ------------
 * 
 * The underscore "_" tags "stream time".  Stream timestamps are timestamps on each recorded frame.. they can be far in the past or in the future.
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
 * Different actions for file streams:
 * 
 * Action     | What it does
 * ---------- | ------------------------------------------
 * register   | ..
 * seek       | set reftime, consume frames 'till t==t_
 * play       | set reftime, keep consuming frames
 * stop       | ..
 * deregister | ..
 * 
 * 
 * 
 * Timing for realtime streams 
 * ---------------------------
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
 

