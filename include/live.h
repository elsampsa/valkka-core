/*
 * live.h : Interface to live555
 * 
 * Copyright 2017 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Valkka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Valkka.  If not, see <http://www.gnu.org/licenses/>. 
 * 
 */

/** 
 *  @file    live.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.2.0 
 *  
 *  @brief Interface to live555
 *
 *  @section DESCRIPTION
 *  
 *  This live555 "bridge" is based on the celebrated "testRTSPClient" test program. 
 * 
 *  Acknowledgements: Ross Finlayson for his advices
 *
 */

#include "livedep.h"
#include "frames.h"
#include "logging.h"

UsageEnvironment& operator<<(UsageEnvironment& env, const RTSPClient& rtspClient);       ///< A function that outputs a string that identifies each stream (for debugging output).
UsageEnvironment& operator<<(UsageEnvironment& env, const MediaSubsession& subsession);  ///< A function that outputs a string that identifies each subsession (for debugging output).
Logger& operator<<(Logger& env, const RTSPClient& rtspClient);                           ///< A function that outputs a string that identifies each stream (for debugging output).
Logger& operator<<(Logger& env, const MediaSubsession& subsession);                      ///< A function that outputs a string that identifies each subsession (for debugging output).
void usage(UsageEnvironment& env, char const* progName);


enum class LiveStatus {
  none,
  alive,
  closed
};


/** Class to hold per-stream state that we maintain throughout each stream's lifetime
 * @ingroup live_tag
 */
class StreamClientState {
public:
  StreamClientState();
  virtual ~StreamClientState();

public:
  MediaSubsessionIterator* iter;
  int subsession_index;
  MediaSession* session;
  MediaSubsession* subsession;
  TaskToken streamTimerTask;
  double duration;
};

/** Handles a live555 RTSP connection
 * 
 * To get an idea how this works, see \ref live555_page
 * 
 * @param env                The usage environment, i.e. event loop in question
 * @param rtspURL            The URL of the live stream
 * @param framefilter        Start of the frame filter chain.  New frames are being fed here.
 * @param verbosityLevel     (optional) Verbosity level
 * @param applicationName    (optional)
 * @param tunnelOverHTTPPortNum (optional)
 * 
 * @ingroup live_tag
 */
class ValkkaRTSPClient: public RTSPClient {
  
public:
  static ValkkaRTSPClient* createNew(UsageEnvironment& env, const std::string rtspURL, FrameFilter& framefilter, LiveStatus* livestatus, int verbosityLevel = 0, char const* applicationName = NULL, portNumBits tunnelOverHTTPPortNum = 0);
  virtual ~ValkkaRTSPClient();
  
protected:
  ValkkaRTSPClient(UsageEnvironment& env, const std::string rtspURL, FrameFilter& framefilter, LiveStatus* livestatus, int verbosityLevel, char const* applicationName, portNumBits tunnelOverHTTPPortNum);
  
public:
  StreamClientState scs;
  FrameFilter& framefilter;
  LiveStatus* livestatus; ///< This points to a variable that is being used by LiveThread
  
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
 * When the event loop has composed a new frame, it's passed to this class and afterGettingFrame is called.  In our case, we pass the new frame to the beginning of a frame filter chain.
 * 
 * @param env        - The usage environment, i.e. event loop in question
 * @param subsession - Identifies the kind of data that's being received (media type, codec, etc.)
 * @param framefilter - FrameFilter to be appliced to the frame.  The start of a FrameFilter callback cascade.
 * @param streamId   - (optional) identifies the stream itself
 * 
 * @ingroup live_tag
 */
class FrameSink: public MediaSink {

public:
  static FrameSink* createNew(UsageEnvironment& env, MediaSubsession& subsession, FrameFilter& framefilter, int subsession_index, char const* streamId = NULL);

private:
  FrameSink(UsageEnvironment& env, MediaSubsession& subsession, FrameFilter& framefilter, int subsession_index, char const* streamId);
    // called only by "createNew()"
  virtual ~FrameSink();

  static void afterGettingFrame(void* clientData, unsigned frameSize, unsigned numTruncatedBytes, struct timeval presentationTime, unsigned durationInMicroseconds);
  void afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes, struct timeval presentationTime, unsigned durationInMicroseconds);
  
private:
  void setReceiveBuffer(unsigned target_size);
  unsigned checkBufferSize(unsigned target_size);
  void sendParameterSets();
  
private:
  // redefined virtual functions:
  virtual Boolean continuePlaying();

private:
  u_int8_t*        fReceiveBuffer;
  long unsigned    nbuf;       ///< Size of bytebuffer
  MediaSubsession& fSubsession;
  char*            fStreamId;
  FrameFilter&     framefilter;
  Frame            setupframe; ///< This frame is used to send subsession information
  Frame            frame;      ///< Data is being copied into this frame
  int              subsession_index;

public: // getters & setters
  uint8_t* getReceiveBuffer() {return fReceiveBuffer;}
  
public:
  bool on;
};



