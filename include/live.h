#ifndef live_HEADER_GUARD
#define live_HEADER_GUARD
/*
 * live.h : Interface to live555
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
 *  @file    live.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.9.0 
 *  @brief Interface to live555
 * 
 *  Acknowledgements: Ross Finlayson for his advice
 * 
 */

#include "livedep.h"
#include "frame.h"
#include "framefifo.h"
#include "framefilter.h"
#include "logging.h"

UsageEnvironment& operator<<(UsageEnvironment& env, const RTSPClient& rtspClient);       ///< A function that outputs a string that identifies each stream (for debugging output).
UsageEnvironment& operator<<(UsageEnvironment& env, const MediaSubsession& subsession);  ///< A function that outputs a string that identifies each subsession (for debugging output).
Logger& operator<<(Logger& logger, const RTSPClient& rtspClient);                        ///< A function that outputs a string that identifies each stream (for debugging output).
Logger& operator<<(Logger& logger, const MediaSubsession& subsession);                   ///< A function that outputs a string that identifies each subsession (for debugging output).
void usage(UsageEnvironment& env, char const* progName);



/** Status for the ValkkaRTSPClient
 * 
 * The problem:
 * 
 * When shutdownStream is called, the ValkkaRTSPClient will be reclaimed/annihilated (within the event loop).  If that has happened, we shouldn't touch it anymore..
 * 
 * If we have called shutdownStream outside the sendDESCRIBECommand etc. callback cascade that's registered in the live555 event loop, then the live555 event loop might try to access an annihilated client
 * 
 * @ingroup live_tag
 */
enum class LiveStatus {
  none,          ///< Client has not been initialized
  pending,       ///< Client's been requested to send the describe command.  This might hang for several reasons: camera offline, internet connection lost, etc.  If we annihilate ValkkaRTSPClient, the callbacks in the event loop might still try to use it..  "pending" connections should be closed only at event loop exit
  alive,         ///< Client has succesfully started playing
  closed         ///< Client has been closed and Medium::close has been called on the MediaSession, etc.  This is done by shutdownStream (which sets livestatus to LiveStatus::closed).  ValkkaRTSPClient has been annihilated!
};


/** Class to hold per-stream state that we maintain throughout each stream's lifetime.
 * 
 * An instance of this class is included in the ValkkaRTSPClient and used in the response handlers / callback chain.
 * 
 * This is a bit cumbersome .. Some of the members are created/managed by the ValkkaRTSPClient instance.  When ValkkaRTSPClient destructs itself (by calling Medium::close on its sinks and MediaSession) in the response-handler callback chain, we need to know that in LiveThread.
 *
 * @ingroup live_tag
 */
class StreamClientState {
public:
  StreamClientState();            ///< Default constructor
  virtual ~StreamClientState();   ///< Default virtual destructor.  Calls Medium::close on the MediaSession object

public:
  MediaSubsessionIterator* iter;  ///< Created by RTSPClient or SDPClient.  Deleted by StreamClientState::~StreamClientState
  int subsession_index;           ///< Managed by RTSPClient or SDPClient
  MediaSession* session;          ///< Created by RTSPClient or SDPClient.  Closed by StreamClientState::~StreamClientState
  MediaSubsession* subsession;    ///< Created by RTSPClient or SDPClient.  Closed by StreamClientState::close
  TaskToken streamTimerTask;
  double duration;
  bool frame_flag;                ///< Set always when a frame is received
  
public:
  void close();                   ///< Calls Medium::close on the MediaSubsession objects and their sinks
  
public: // setters & getters
  void setFrame()     {this->frame_flag=true;}
  void clearFrame()   {this->frame_flag=false;}
  bool gotFrame()     {return this->frame_flag;}
};



/** Handles a live555 RTSP connection
 * 
 * To get an idea how this works, see \ref live555_page
 * 
 * @ingroup live_tag
 */
class ValkkaRTSPClient: public RTSPClient {
  
public:
  /** Default constructor
   * @param env                   The usage environment, i.e. event loop in question
   * @param rtspURL               The URL of the live stream
   * @param framefilter           Start of the frame filter chain.  New frames are being fed here.
   * @param livestatus            This used to inform LiveThread about the state of the stream
   * @param verbosityLevel        (optional) Verbosity level
   * @param applicationName       (optional)
   * @param tunnelOverHTTPPortNum (optional)
   */
  static ValkkaRTSPClient* createNew(UsageEnvironment& env, const std::string rtspURL, FrameFilter& framefilter, LiveStatus* livestatus, int verbosityLevel = 0, char const* applicationName = NULL, portNumBits tunnelOverHTTPPortNum = 0);
  
protected:
  ValkkaRTSPClient(UsageEnvironment& env, const std::string rtspURL, FrameFilter& framefilter, LiveStatus* livestatus, int verbosityLevel, char const* applicationName, portNumBits tunnelOverHTTPPortNum);
  /** Default virtual destructor */
  virtual ~ValkkaRTSPClient();
  
public:
  StreamClientState scs;
  FrameFilter& framefilter;     ///< Target frame filter where frames are being fed
  LiveStatus* livestatus;       ///< This points to a variable that is being used by LiveThread to inform about the stream state
  
public: // some extra parameters and their setters
  bool     request_multicast; ///< Request multicast during rtsp negotiation
  bool     request_tcp;       ///< Request interleaved streaming over tcp
  unsigned recv_buffer_size;  ///< Operating system ringbuffer size for incoming socket
  unsigned reordering_time;   ///< Live555 packet reordering treshold time (microsecs)
  void     requestMulticast()               {this->request_multicast=true;}
  void     requestTCP()                     {this->request_tcp=true;}
  void     setRecvBufferSize(unsigned i)    {this->recv_buffer_size=i;}
  void     setReorderingTime(unsigned i)    {this->reordering_time=i;}
  
public: 
  // Response handlers
  static void continueAfterDESCRIBE(RTSPClient* rtspClient, int resultCode, char* resultString); ///< Called after rtsp DESCRIBE command gets a reply
  static void continueAfterSETUP(RTSPClient* rtspClient, int resultCode, char* resultString);    ///< Called after rtsp SETUP command gets a reply
  static void continueAfterPLAY(RTSPClient* rtspClient, int resultCode, char* resultString);     ///< Called after rtsp PLAY command gets a reply

  // Other event handler functions:
  static void subsessionAfterPlaying(void* clientData); ///< Called when a stream's subsession (e.g., audio or video substream) ends
  static void subsessionByeHandler(void* clientData);   ///< Called when a RTCP "BYE" is received for a subsession
  static void streamTimerHandler(void* clientData);     ///< Called at the end of a stream's expected duration (if the stream has not already signaled its end using a RTCP "BYE")
  static void setupNextSubsession(RTSPClient* rtspClient); ///< Used to iterate through each stream's 'subsessions', setting up each one
  static void shutdownStream(RTSPClient* rtspClient, int exitCode = 1); ///< Used to shut down and close a stream (including its "RTSPClient" object):
  
};


/** Live555 handling of media frames 
 * 
 * When the live555 event loop has composed a new frame, it's passed to FrameSink::afterGettingFrame.  There it is passed to the beginning of valkka framefilter chain.
 * 
 * @ingroup live_tag
 */
class FrameSink: public MediaSink {

public:

  /** Default constructor
   * @param env          The usage environment, i.e. event loop in question
   * @param scs          Info about the stream state
   * @param subsession   Identifies the kind of data that's being received (media type, codec, etc.)
   * @param framefilter  The start of valkka FrameFilter filterchain
   * @param streamId     (optional) identifies the stream itself
   * 
   */
  static FrameSink* createNew(UsageEnvironment& env, StreamClientState& scs, FrameFilter& framefilter, char const* streamId = NULL);

private:
  FrameSink(UsageEnvironment& env, StreamClientState& scs, FrameFilter& framefilter, char const* streamId);
  /** Default virtual destructor */
  virtual ~FrameSink();

  static void afterGettingFrame(void* clientData, unsigned frameSize, unsigned numTruncatedBytes, struct timeval presentationTime, unsigned durationInMicroseconds); ///< Called after live555 event loop has composed a new frame
  void afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes, struct timeval presentationTime, unsigned durationInMicroseconds); ///< Called by the other afterGettingFrame
  
private:
  void setReceiveBuffer(unsigned target_size);     ///< Calculates receiving memory buffer size
  unsigned checkBufferSize(unsigned target_size);  ///< Calculates receiving memory buffer size
  void sendParameterSets();                        ///< Extracts sps and pps info from the SDP string.  Creates sps and pps frames and sends them to the filterchain.
  
private: // redefined virtual functions:
  virtual Boolean continuePlaying();  ///< Live555 redefined virtual function

private:
  StreamClientState &scs;
  u_int8_t*         fReceiveBuffer;
  long unsigned     nbuf;       ///< Size of bytebuffer
  MediaSubsession&  fSubsession;
  char*             fStreamId;
  FrameFilter&      framefilter;
  SetupFrame        setupframe;  ///< This frame is used to send subsession information
  BasicFrame        basicframe;  ///< Data is being copied into this frame
  int               subsession_index;

public: // getters & setters
  uint8_t* getReceiveBuffer() {return fReceiveBuffer;}
  
public:
  bool on;
};

#endif
