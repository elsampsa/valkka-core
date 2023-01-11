#ifndef LIVETHREAD_HEADER_GUARD 
#define LIVETHREAD_HEADER_GUARD
/*
 * livethread.h : A live555 thread
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
 *  @file    livethread.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.3.4 
 *  
 *  @brief A live555 thread
 *
 */

#include "live.h"
#include "liveserver.h"
#include "thread.h"
#include "framefilter.h"
#include "framefifo.h"
#include "event.h"


void setLiveOutPacketBuffermaxSize(unsigned i); // <pyapi>


/** This is a special FrameFifo class for feeding frames *into* live555, i.e. for sending them to the network.
 * 
 * Should not be instantiated by the user, but requested from LiveThread with LiveThread::getFifo()
 * 
 * There is a single LiveFifo instance per LiveThread
 * 
 * @ingroup livethread_tag
 * @ingroup queues_tag
 */
class LiveFifo : public FrameFifo {                       
  
public:                                                    
  /** Default constructor */
  LiveFifo(const char* name, FrameFifoContext ctx);       
  /** Default virtual destructor */
  ~LiveFifo();                                            
  
protected:
  void* live_thread;
  
public:
  void setLiveThread(void* live_thread);
  bool writeCopy(Frame* f, bool wait=false);
};                                                        


/** LiveThread connection types
 * 
 * Identifies different kinds of connections (i.e. rtsp and sdp).  Used by LiveConnectionContext.
 * 
 * @ingroup livethread_tag
 */
enum class LiveConnectionType { // <pyapi>
  none,                         // <pyapi>
  rtsp,                         // <pyapi>
  sdp                           // <pyapi>
};                              // <pyapi>


/** Identifies a stream and encapsulates information about the type of connection the user is requesting to LiveThread.  LiveConnectionContext is also included in LiveThread::SignalContext, i.e. it carries the signal information to LiveThread (for the thread signaling system, see \ref threading_tag).
  * 
  * (A side note: this class is not nested inside the LiveThread class for one simple reason: swig does not like nested classes, so it would make it harder to create Python bindings)
  * 
  * Information in LiveConnectionContext is further passed by LiveThread to RTSPConnection and SDPConnection
  * 
  * Comes with two different versions of the constructor.  First is for primary use and the second is a "dummy" constructor.
  * 
  * @ingroup livethread_tag
  */  
struct LiveConnectionContext {                                                                        // <pyapi>
  /** Default constructor */
  LiveConnectionContext(LiveConnectionType ct, std::string address, SlotNumber slot,                  // <pyapi>
                        FrameFilter* framefilter) :                                                   // <pyapi>
  connection_type(ct), address(address), slot(slot), framefilter(framefilter), msreconnect(10000),       // <pyapi>
  request_multicast(false), request_tcp(false), recv_buffer_size(0), reordering_time(0),              // <pyapi>
  time_correction(TimeCorrectionType::smart)                                                          // <pyapi>
  {}                                                                                                  // <pyapi>
  /** Dummy constructor : remember to set member values by hand */
  LiveConnectionContext() :                                                                           // <pyapi>
  connection_type(LiveConnectionType::none), address(""), slot(0), framefilter(NULL), msreconnect(10000), // <pyapi>
  request_multicast(false), request_tcp(false),time_correction(TimeCorrectionType::smart)             // <pyapi>
  {}                                                                                                  // <pyapi>
  LiveConnectionType connection_type;   ///< Identifies the connection type                           // <pyapi>
  std::string        address;           ///< Stream address                                           // <pyapi>
  SlotNumber         slot;              ///< A unique stream slot that identifies this stream         // <pyapi>
  FrameFilter*       framefilter;       ///< The frames are feeded into this FrameFilter              // <pyapi>
  long unsigned int  msreconnect;       ///< If stream has delivered nothing during this many milliseconds, reconnect // <pyapi>
  bool               request_multicast; ///< Request multicast in the rtsp negotiation or not         // <pyapi>
  bool               request_tcp;       ///< Request interleaved rtsp streaming or not                // <pyapi>
  unsigned           recv_buffer_size;  ///< Operating system ringbuffer size for incoming socket     // <pyapi>
  unsigned           reordering_time;   ///< Live555 packet reordering treshold time (microsecs)      // <pyapi>
  TimeCorrectionType time_correction;   ///< How to perform frame timestamp correction                // <pyapi>
};                                                                                                    // <pyapi>


/** Same as LiveConnectionContext, but for outbound streams (i.e. streams sent over the net by live555)
 * 
 * @ingroup livethread_tag
 */
struct LiveOutboundContext {                                                                     // <pyapi>
  LiveOutboundContext(LiveConnectionType ct, std::string address, SlotNumber slot,               // <pyapi>
                      unsigned short int portnum) :                                              // <pyapi>
  connection_type(ct), address(address), slot(slot), portnum(portnum), ttl(225)                  // <pyapi>
  {}                                                                                             // <pyapi>
  LiveOutboundContext() :                                                                        // <pyapi>
  connection_type(LiveConnectionType::none), address(""), slot(0), portnum(0), ttl(255)          // <pyapi>
  {}                                                                                             // <pyapi>
  LiveConnectionType  connection_type; ///< Identifies the connection type                       // <pyapi>
  std::string         address;         ///< Stream address                                       // <pyapi>
  SlotNumber          slot;            ///< A unique stream slot that identifies this stream     // <pyapi>
  unsigned short int  portnum;         ///< Start port number (for sdp)                          // <pyapi>
  unsigned char       ttl;             ///< Packet time-to-live                                  // <pyapi>
};                                                                                               // <pyapi>



/** Characteristic signals for the Live555 thread.
* 
* These signals map directly into methods with the same names
* 
*/
enum class LiveSignal {
  none,
  exit,
  // inbound streams
  register_stream,
  deregister_stream,
  play_stream,
  stop_stream,
  // outbound streams
  register_outbound,
  deregister_outbound
};


/** Identifies the information the signals LiveThread::Signals carry.  Encapsulates a LiveConnectionContext and a LiveOutboundContext instance (one of the two is used, depending on the signal)
*
*/
struct LiveSignalContext {
  LiveSignal              signal;
  LiveConnectionContext   *connection_context;
  LiveOutboundContext     *outbound_context;
};



/** A base class that unifies all kinds of connections (RTSP and SDP).
 * 
 * Methods of this class are used by the LiveThread class and they are called from within the Live555 event loop.
 * 
 * Connect typically has a small, default internal filterchain to correct for the often-so-erroneous timestamps (see the cpp file for more details):
 * 
 * Filterchain: --> {FrameFilter: Connection::inputfilter} --> {TimestampFrameFilter2: Connection::timestampfilter} --> {FrameFilter: Connection::framefilter} -->
 * 
 * @ingroup livethread_tag
 * 
 */ 
class Connection {
  
public:
  /** Default constructor
   * 
   * @param env   See Connection::env
   * @param ctx   See Connection::ctx
   * 
   * @ingroup livethread_tag
   */
  Connection(UsageEnvironment& env, LiveConnectionContext& ctx);
  virtual ~Connection(); ///< Default destructor
  
protected:
  LiveConnectionContext   &ctx;           ///< LiveConnectionContext identifying the stream source (address), it's destination (slot and target framefilter), etc. 
  
  // internal framefilter chain.. if we'd like to modify the frames before they are passed to the API user
  // more framefilters could be generated here and initialized in the constructor init list
  // the starting filter should always be named as "inputfilter" .. this is where Live555 writes the frames
  // TimestampFrameFilter2   timestampfilter; ///< Internal framefilter: correct timestamp
  // SlotFrameFilter         inputfilter;     ///< Internal framefilter: set slot number
  
  FrameFilter*            timestampfilter;
  FrameFilter*            inputfilter;
  FrameFilter*            repeat_sps_filter;        ///< Repeat sps & pps packets before i-frame (if they were not there before the i-frame)
  
  long int                frametimer;      ///< Measures time when the last frame was received
  long int                pendingtimer;    ///< Measures how long stream has been pending
  
public:
  UsageEnvironment &env;                   ///< UsageEnvironment identifying the Live555 event loop (see \ref live555_page)
  bool is_playing;
  
public:
  virtual void playStream() =0;   ///< Called from within the live555 event loop
  virtual void stopStream() =0;   ///< Stops stream and reclaims it resources.  Called from within the live555 event loop
  virtual void reStartStream();   ///< Called from within the live555 event loop
  virtual void reStartStreamIf(); ///< Called from within the live555 event loop
  virtual bool isClosed();        ///< Have the streams resources been reclaimed after stopping it?
  virtual void forceClose();      ///< Normally, stopStream reclaims the resources.  This one forces the delete.
  SlotNumber getSlot();           ///< Return the slot number
};


/** A base class that unifies all kinds of outgoing streams (i.e. streams sent by live555).  Analogical to Connection (that is for incoming streams).
 * 
 * @param env    See Outbound::env
 * @param fifo   See Outbound::fifo
 * @param ctx    See Outbound::ctx
 * 
 * @ingroup livethread_tag
 */
class Outbound { // will leave this quite generic .. don't know at this point how the rtsp server is going to be // analogy: AVThread
  
public:
  Outbound(UsageEnvironment& env, FrameFifo& fifo, LiveOutboundContext& ctx);  ///< Default constructor
  virtual ~Outbound(); ///< Default virtual destructor
  
public: // init'd at constructor time
  LiveOutboundContext  &ctx;     ///< Identifies the connection type, stream address, etc.  See LiveOutboundContext
  UsageEnvironment     &env;     ///< Identifies the live555 event loop
  FrameFifo            &fifo;    ///< Outgoing fFrames are being read and finally recycled here
  
protected:
  bool setup_ok, at_setup; ///< Flags used by Outbound::handleFrame
  
public:
  virtual void reinit();              ///< Reset session and subsessions
  virtual void handleFrame(Frame *f); ///< Setup session and subsessions, writes payload
};


/** A negotiated RTSP connection
 * 
 * Uses the internal ValkkaRTSPClient instance which defines the RTSP client behaviour, i. e. the events and callbacks that are registered into the Live555 event loop (see \ref live_tag)
 * 
 * @ingroup livethread_tag
 */
class RTSPConnection : public Connection {

public:
  /** @copydoc Connection::Connection */
  RTSPConnection(UsageEnvironment& env, LiveConnectionContext& ctx);
  ~RTSPConnection();
  // RTSPConnection(const RTSPConnection& cp); ///< Copy constructor .. nopes, default copy constructor good enough
  
  
private:
  ValkkaRTSPClient* client; ///< ValkkaRTSPClient defines the behaviour (i.e. event registration and callbacks) of the RTSP client (see \ref live_tag)
  LiveStatus livestatus;    ///< Reference of this variable is passed to ValkkaRTSPClient.  We can see outside of the live555 callback chains if RTSPConnection::client has deallocated itself
  bool termplease;          ///< Ref of this var is passed to ValkkaRTSPClient.  When set to true, ValkkaRTSPClient should terminate itself if not yet playing
  
public:
  void playStream();      ///< Uses ValkkaRTSPClient instance to initiate the RTSP negotiation
  void stopStream();      ///< Uses ValkkaRTSPClient instance to shut down the stream
  void reStartStreamIf(); ///< Restarts the stream if no frames have been received for a while
  bool isClosed();        ///< Have the streams resources been reclaimed?
  void forceClose();
};


/** Connection is is defined in an SDP file
 * 
 * @ingroup livethread_tag
 */
class SDPConnection : public Connection {

public:
  /** @copydoc Connection::Connection */
  SDPConnection(UsageEnvironment& env, LiveConnectionContext& ctx);
  /** Default destructor */
  ~SDPConnection();

private:
  StreamClientState *scs;
  
public:
  void playStream(); ///< Creates Live555 MediaSessions, MediaSinks, etc. instances and registers them directly to the Live555 event loop
  void stopStream(); ///< Closes Live555 MediaSessions, MediaSinks, etc.

};


/** Sending a stream without rtsp negotiation (i.e. without rtsp server) to certain ports
 * 
 * @param env    See Outbound::env
 * @param fifo   See Outbound::fifo
 * @param ctx    See Outbound::ctx
 * 
 * @ingroup livethread_tag
 */
class SDPOutbound : public Outbound {
  
public: 
  SDPOutbound(UsageEnvironment &env, FrameFifo &fifo, LiveOutboundContext& ctx);
  ~SDPOutbound();  
  
public: // virtual redefined
  void reinit();
  void handleFrame(Frame *f);
  
public:
  std::vector<Stream*> streams;  ///< SubStreams of the outgoing streams (typically two, e.g. video and sound)
};


/** Sending a stream using the on-demand rtsp server
 * 
 * @param env    See Outbound::env
 * @param fifo   See Outbound::fifo
 * @param ctx    See Outbound::ctx
 * 
 * @ingroup livethread_tag
 */
class RTSPOutbound : public Outbound {
  
public: 
  RTSPOutbound(UsageEnvironment &env, RTSPServer &server, FrameFifo &fifo, LiveOutboundContext& ctx);
  ~RTSPOutbound();  
  
protected: // init'd at constructor time
  RTSPServer &server;   ///< Reference to the RTSPServer instance
  
public: // virtual redefined
  void reinit();
  void handleFrame(Frame *f);
  
public:
  ServerMediaSession *media_session; 
  std::vector<ValkkaServerMediaSubsession*> media_subsessions;
  
};



/** Live555, running in a separate thread
 * 
 * This class implements a "producer" thread that outputs frames into a FrameFilter (see \ref threading_tag)
 * 
 * This Thread has its own running Live555 event loop.  It registers a callback into the Live555 event loop which checks periodically for signals send to the thread.  Signals to this thread are sent using the LiveThread::sendSignal method.
 * 
 * API methods take as parameter either LiveConnectionContext or LiveOutboundContext instances that identify the stream (type, address, slot number, etc.)
 *
 * @ingroup livethread_tag
 * @ingroup threading_tag
 */  
class LiveThread : public Thread { // <pyapi>
  
public:                           
  static void periodicTask(void* cdata); ///< Used to (re)schedule LiveThread methods into the live555 event loop

public:                                                // <pyapi>
  /** Default constructor
   * 
   * @param name          Thread name
   * @param n_max_slots   Maximum number of connections (each Connection instance is placed in a slot)
   * 
   */
  LiveThread(const char* name, FrameFifoContext fifo_ctx=FrameFifoContext());  // <pyapi>
  ~LiveThread();                                                               // <pyapi>
  
protected: // frame input
  LiveFifo          infifo;     ///< A FrameFifo for incoming frames
  FifoFrameFilter   infilter;   ///< A FrameFilter for writing incoming frames
  
protected: // redefinitions
  std::deque<LiveSignalContext> signal_fifo;    ///< Redefinition of signal fifo (Thread::signal_fifo becomes hidden)
  
protected:
  TaskScheduler*    scheduler;               ///< Live555 event loop TaskScheduler
  UsageEnvironment* env;                     ///< Live555 UsageEnvironment identifying the event loop
  char              eventLoopWatchVariable;  ///< Modifying this, kills the Live555 event loop
  std::vector<Connection*>   slots_;         ///< A constant sized vector.  Book-keeping of the connections (RTSP or SDP) currently active in the live555 thread.  Organized in "slots".
  std::vector<Outbound*>     out_slots_;     ///< Book-keeping for the outbound connections
  std::list<Connection*>     pending;        ///< Incoming connections pending for closing
  bool                       exit_requested; ///< Exit asap
  EventTriggerId    event_trigger_id_hello_world;
  EventTriggerId    event_trigger_id_frame_arrived;
  EventTriggerId    event_trigger_id_got_frames;
  int fc;                                    ///< debugging: incoming frame counter
  
protected: // rtsp server for live and/or recorded stream
  UserAuthenticationDatabase  *authDB;
  RTSPServer                  *server;
  
public: // redefined virtual functions
  void run();
  void preRun();
  void postRun();
  /** @copydoc Thread::sendSignal */
  void sendSignal(LiveSignalContext signal_ctx);
  
protected:
  void handlePending();       ///< Try to close streams that were not properly closed (i.e. idling for the tcp socket while closing).  Used by LiveThread::periodicTask
  void checkAlive();          ///< Used by LiveThread::periodicTask
  void closePending();        ///< Force close all pending connections
  void handleSignals();       ///< Handle pending signals in the signals queue.  Used by LiveThread::periodicTask
  void handleFrame(Frame* f); ///< Handle incoming frames.  See \ref live_streaming_page
  
private: // internal
  int  safeGetSlot         (SlotNumber slot, Connection*& con);
  int  safeGetOutboundSlot (SlotNumber slot, Outbound*& outbound);
  // inbound streams
  void registerStream   (LiveConnectionContext &connection_ctx);
  void deregisterStream (LiveConnectionContext &connection_ctx);
  void playStream       (LiveConnectionContext &connection_ctx);
  // outbound streams
  void registerOutbound    (LiveOutboundContext &outbound_ctx); 
  void deregisterOutbound  (LiveOutboundContext &outbound_ctx);
  // thread control
  void stopStream       (LiveConnectionContext &connection_ctx);
  
public: // *** C & Python API *** .. these routines go through the condvar/mutex locking                                                // <pyapi>
  // inbound streams
  void registerStreamCall   (LiveConnectionContext &connection_ctx); ///< API method: registers a stream                                // <pyapi> 
  void deregisterStreamCall (LiveConnectionContext &connection_ctx); ///< API method: de-registers a stream                             // <pyapi>
  void playStreamCall       (LiveConnectionContext &connection_ctx); ///< API method: starts playing the stream and feeding frames      // <pyapi>
  void stopStreamCall       (LiveConnectionContext &connection_ctx); ///< API method: stops playing the stream and feeding frames       // <pyapi>
  // outbound streams
  void registerOutboundCall   (LiveOutboundContext &outbound_ctx);   ///< API method: register outbound stream                          // <pyapi>
  void deregisterOutboundCall (LiveOutboundContext &outbound_ctx);   ///< API method: deregister outbound stream                        // <pyapi>
  // thread control
  void requestStopCall();                                            ///< API method: Like Thread::stopCall() but does not block        // <pyapi>
  // LiveFifo &getFifo();                                            ///< API method: get fifo for sending frames with live555          // <pyapi>
  FifoFrameFilter &getFrameFilter();                                 ///< API method: get filter for sending frames with live555        // <pyapi>
  void setRTSPServer(int portnum=8554);                              ///< API method: activate the RTSP server at port portnum          // <pyapi>
  virtual void waitReady();                                          ///< API method: wait until all signals and pending connections are resolved  // <pyapi>

public: // live555 events and tasks
  static void helloWorldEvent(void* clientData);   ///< For testing/debugging  
  static void frameArrivedEvent(void* clientData); ///< For debugging
  static void gotFramesEvent(void* clientData);    ///< Triggered when an empty fifo gets a frame.  Schedules readFrameFifoTask.  See \ref live_streaming_page
  static void readFrameFifoTask(void* clientData); ///< This task registers itself if there are frames in the fifo

public:  
  void testTrigger();         ///< See \ref live_streaming_page
  void triggerGotFrames();    ///< See \ref live_streaming_page
}; // <pyapi>

#endif
