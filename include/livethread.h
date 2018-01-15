#ifndef LIVETHREAD_HEADER_GUARD 
#define LIVETHREAD_HEADER_GUARD
/*
 * livethread.h : A live555 thread
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
 *  @file    livethread.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.2.0 
 *  
 *  @brief A live555 thread
 *
 */

#include "live.h"
#include "threads.h"
#include "filters.h"


/** A base class that unifies all kinds of connections (RTSP and SDP).
 * 
 * Methods of this class are used by the LiveThread class and they are called from within the Live555 event loop (i.e. this is not part of the API)
 * 
 * @ingroup livethread_tag
 * @ingroup live_tag
 * 
 *  Has an internal filterchain:
 * 
 *  Filterchain: --> {FrameFilter: Connection::inputfilter} --> {TimestampFrameFilter: Connection::timestampfilter} --> {FrameFilter: Connection::framefilter} -->
 * 
 */ 
class Connection {
  
public:
  /** 
   * @param env          UsageEnvironment identifying the Live555 event loop (see \ref live555_page)
   * @param address      A string identifying the connection
   * @param slot         An integer defining a "slot" where this connection is placed
   * @param framefilter  Connection feeds frames to this FrameFilter (i.e., its the beginning of the "filter-chain")
   */
  Connection(UsageEnvironment& env, std::string address, SlotNumber slot, FrameFilter& framefilter);
  virtual ~Connection(); ///< Default destructor
  
protected:
  std::string         address;      ///< Stream address
  SlotNumber          slot;         ///< Stream slot number (that identifies the source)
  FrameFilter&        framefilter;  ///< User-provided entry point for the stream. 
  // internal framefilter chain.. if we'd like to modify the frames before they are passed to the API user
  // more framefilter could be generated here, initialized it the constructor init list
  // the starting filter should always be named as "inputfilter" .. this is where Live555 writes the frames
  TimestampFrameFilter    timestampfilter; ///< Internal framefilter: correct timestamp
  SlotFrameFilter         inputfilter;     ///< Internal framefilter: set slot number
  
public:
  UsageEnvironment& env;
  bool is_playing;
  
public:
  virtual void playStream() =0;   ///< Called from within the live555 event loop
  virtual void stopStream() =0;   ///< Called from within the live555 event loop
  virtual void reStartStream();   ///< Called from within the live555 event loop
  SlotNumber getSlot();           ///< Return the slot number
};


/** A negotiated RTSP connection
 * 
 * Uses the internal ValkkaRTSPClient instance which defines the RTSP client behaviour, i. e. the events and callbacks that are registered into the Live555 event loop (see \ref live_tag)
 * 
 * @ingroup livethread_tag
 * @ingroup live_tag
 */
class RTSPConnection : public Connection {

public:
  /** @copydoc Connection::Connection */
  RTSPConnection(UsageEnvironment& env, const std::string address, SlotNumber slot, FrameFilter& framefilter); 
  ~RTSPConnection();

private:
  ValkkaRTSPClient* client; ///< ValkkaRTSPClient defines the behaviour (i.e. event registration and callbacks) of the RTSP client (see \ref live_tag)
  LiveStatus livestatus;
  
public:
  void playStream(); ///< Uses ValkkaRTSPClient instance to initiate the RTSP negotiation
  void stopStream(); ///< Uses ValkkaRTSPClient instance to shut down the stream
  
};


/** Streaming, the source of stream is defined in an SDP file
 * 
 * @ingroup livethread_tag
 * @ingroup live_tag
 */
class SDPConnection : public Connection {

public:
  /** @copydoc Connection::Connection */
  SDPConnection(UsageEnvironment& env, const std::string address, SlotNumber slot, FrameFilter& framefilter);
  ~SDPConnection();

private:
  MediaSession* session;
  
public:
  void playStream(); ///< Creates Live555 MediaSessions, MediaSinks, etc. instances and registers them directly to the Live555 event loop
  void stopStream(); ///< Closes Live555 MediaSessions, MediaSinks, etc.

};


/** LiveThread connection types
 * 
 * This enumeration class identifies different kinds of connections (i.e. rtsp and sdp).  Used by LiveConnectionContext.
 * 
 * @ingroup livethread_tag
 * @ingroup threading_tag
 * @ingroup live_tag
 */
enum class LiveConnectionType { // <pyapi>
  none,                         // <pyapi>
  rtsp,                         // <pyapi>
  sdp                           // <pyapi>
};                              // <pyapi>

/** Identifies a stream and encapsulates information about the type of connection, the user is requesting to LiveThread.  LiveConnectionContext is included into LiveThread::SignalContext, i.e. it carries the signal information to LiveThread.  For the thread signaling system, see \ref threading_tag
  * 
  * (A side note: this class is not nested inside the LiveThread class for one simple reason: swig does not like nested classes, so it would make it harder to create Python bindings)
  * 
  * Information in LiveConnectionContext is passed by LiveThread to RTSPConnection and SDPConnection
  * 
  * @ingroup livethread_tag
  * @ingroup threading_tag
  * @ingroup live_tag
  */  
struct LiveConnectionContext { // <pyapi>
  LiveConnectionType connection_type; ///< Identifies the connection type                    // <pyapi>
  std::string        address;         ///< Stream address                                    // <pyapi>
  SlotNumber         slot;            ///< A unique stream slot that identifies this stream  // <pyapi>
  FrameFilter*       framefilter;     ///< The frames are feeded into this FrameFilter       // <pyapi>
  // LiveConnectionContext() : connection_type(ConnectionType::none), address(""), slot(0), framefilter(NULL) {} // Initialization to a default value : does not compile ..! // <pyapi>
};                             // <pyapi>


/** Live555, running in a separate thread
 * 
 * This class implements a "producer" thread that outputs frames into a FrameFilter (see \ref threading_tag)
 * 
 * This Thread has its own running Live555 event loop.  At constructor time, it registers a callback into the Live555 event loop which checks periodically for signals the API user has been sending, by using the API methods.  The API methods are: registerStream, deregisterStream, playStream and stopStream
 * 
 * The API methods use, internally, the sendSignal method for thread-safe communication with LiveThread
 * 
 * The API methods take as parameter, a LiveConnectionContext instance identifying the stream (type, address, slot number, etc.)
 *
 * @ingroup livethread_tag
 * @ingroup threading_tag
 * @ingroup live_tag
 */  
class LiveThread : public Thread { // <pyapi>
  
public:                           
  
  /** Characteristic signals for the Live555 thread.
   * 
   * These signals map directly into methods with the same names
   * 
   */
  enum class Signals {
    none,
    exit,                 
    register_stream,
    deregister_stream,
    play_stream,
    stop_stream
  };
  
  /** Identifies the information the signals LiveThread::Signals carry.  Encapsulates a LiveConnectionContext instance.
   *
   */
  struct SignalContext {
    Signals                 signal;
    LiveConnectionContext   *connection_context;
  };

  static void periodicTask(void* cdata); ///< Used to (re)schedule LiveThread methods into the live555 event loop

public:                                                // <pyapi>
  /** Default constructor
   * 
   * @param name          Thread name
   * @param n_max_slots   Maximum number of connections (each Connection instance is placed in a slot)
   * 
   */
  LiveThread(const char* name, int core_id=-1);        // <pyapi>
  ~LiveThread();                                       // <pyapi>
  
protected: // redefinitions
  std::deque<SignalContext> signal_fifo;    ///< Redefinition of signal fifo (Thread::signal_fifo is now hidden from usage) 
  
protected:
  TaskScheduler*    scheduler;              ///< Live555 event loop TaskScheduler
  UsageEnvironment* env;                    ///< Live555 UsageEnvironment identifying the event loop
  char              eventLoopWatchVariable; ///< Modifying this, kills the Live555 event loop
  std::vector<Connection*>   slots_;        ///< A constant sized vector.  Book-keeping of the connections (RTSP or SDP) currently active in the live555 thread.  Organized in "slots".
  // SlotNumber n_max_slots;                   ///< Maximum number of possible slots .. use a global constant (in sizes.h)
  
public: // redefined virtual functions
  void run();
  void preRun();
  void postRun();
  /** @copydoc Thread::sendSignal */
  void sendSignal(SignalContext signal_ctx); ///< Must be explicitly *redefined* just in case : Thread::SignalContext has been changed to LiveThread::SignalContext
  
protected:
  void handleSignals();
  
private: // internal
  int  safeGetSlot      (SlotNumber slot, Connection*& con);
  void registerStream   (LiveConnectionContext &connection_ctx);
  void deregisterStream (LiveConnectionContext &connection_ctx);
  void playStream       (LiveConnectionContext &connection_ctx);
  void stopStream       (LiveConnectionContext &connection_ctx);
  
public: // *** C & Python API *** .. these routines go through the convar/mutex locking                                                // <pyapi>
  void registerStreamCall   (LiveConnectionContext &connection_ctx); ///< API method: registers a stream                                // <pyapi> 
  void deregisterStreamCall (LiveConnectionContext &connection_ctx); ///< API method: de-registers a stream                             // <pyapi>
  void playStreamCall       (LiveConnectionContext &connection_ctx); ///< API method: starts playing the stream and feeding frames      // <pyapi>
  void stopStreamCall       (LiveConnectionContext &connection_ctx); ///< API method: stops playing the stream and feeding frames       // <pyapi>
  void stopCall();                                                  ///< API method: stops the LiveThread                              // <pyapi>
}; // <pyapi>

#endif