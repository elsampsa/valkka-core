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
 *  @version 0.3.0 
 *  
 *  @brief Interface to live555
 * 
 *  Acknowledgements: Ross Finlayson for his advice
 *
 */

#include "livedep.h"
#include "frames.h"
#include "queues.h"
#include "logging.h"

UsageEnvironment& operator<<(UsageEnvironment& env, const RTSPClient& rtspClient);       ///< A function that outputs a string that identifies each stream (for debugging output).
UsageEnvironment& operator<<(UsageEnvironment& env, const MediaSubsession& subsession);  ///< A function that outputs a string that identifies each subsession (for debugging output).
Logger& operator<<(Logger& logger, const RTSPClient& rtspClient);                           ///< A function that outputs a string that identifies each stream (for debugging output).
Logger& operator<<(Logger& logger, const MediaSubsession& subsession);                      ///< A function that outputs a string that identifies each subsession (for debugging output).
void usage(UsageEnvironment& env, char const* progName);



class BufferSource: public FramedSource {

public:
  BufferSource(UsageEnvironment &env, FrameFifo &fifo, unsigned preferredFrameSize =0, unsigned playTimePerFrame =0, unsigned offset=0);
  virtual ~BufferSource();
  
private:
  virtual void doGetNextFrame();
  
private:
  FrameFifo          &fifo;
  
  // uint8_t*  fBuffer;
  // unsigned  fMstimestamp;
  // unsigned  fBufferSize;
  
  // Boolean   fDeleteBufferOnClose;
  unsigned  fPreferredFrameSize;
  unsigned  fPlayTimePerFrame;
  unsigned  offset;
  // unsigned  fLastPlayTime;
  // Boolean   fLimitNumBytesToStream;
  // u_int64_t fNumBytesToStream; // used if "fLimitNumBytesToStream" is True
  
public:
  std::deque<Frame*> internal_fifo;
  bool      active;
  
public:
  void handleFrame(Frame* f);
  
};


// encapsulates an outbound stream
class Stream { // analogy: DecoderBase
  
public:
  Stream(UsageEnvironment &env, FrameFifo &fifo, const std::string adr, unsigned short int portnum, const unsigned char ttl=255);
  virtual ~Stream();
  
protected:
  UsageEnvironment  &env;
  FrameFifo         &fifo;
  
  RTPSink           *sink; // queries frames from terminal
  RTCPInstance      *rtcp;
  
  Groupsock  *rtpGroupsock;
  Groupsock  *rtcpGroupsock;
  unsigned char cname[101];
  
  BufferSource *buffer_source;
  FramedSource *terminal; // the final device in the live555 filterchain
  
public:
  void handleFrame(Frame *f);
  void startPlaying();
  static void afterPlaying(void* cdata);
  
};


class H264Stream : public Stream {
 
public:
  H264Stream(UsageEnvironment &env, FrameFifo &fifo, const std::string adr, unsigned short int portnum, const unsigned char ttl=255);
  ~H264Stream();
    
};



enum class LiveStatus {
  none,
  pending,
  alive,
  closed
};


/** Class to hold per-stream state that we maintain throughout each stream's lifetime
 * @ingroup live_tag
 */
class StreamClientState {
public:
  StreamClientState();
  virtual ~StreamClientState();   ///< Calls Medium::close on the MediaSession object

public:
  MediaSubsessionIterator* iter;  ///< Created by RTSPClient or SDPClient.  Deleted by StreamClientState::~StreamClientState
  int subsession_index;           ///< Managed by RTSPClient or SDPClient
  MediaSession* session;          ///< Created by RTSPClient or SDPClient.  Closed by StreamClientState::~StreamClientState
  MediaSubsession* subsession;    ///< Created by RTSPClient or SDPClient.  Closed by StreamClientState::close
  TaskToken streamTimerTask;
  double duration;
  bool frame_flag;
  
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
  // static FrameSink* createNew(UsageEnvironment& env, MediaSubsession& subsession, FrameFilter& framefilter, int subsession_index, char const* streamId = NULL);
  static FrameSink* createNew(UsageEnvironment& env, StreamClientState& scs, FrameFilter& framefilter, char const* streamId = NULL);

  
private:
  // FrameSink(UsageEnvironment& env, MediaSubsession& subsession, FrameFilter& framefilter, int subsession_index, char const* streamId);
  FrameSink(UsageEnvironment& env, StreamClientState& scs, FrameFilter& framefilter, char const* streamId);
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
  StreamClientState &scs;
  u_int8_t*         fReceiveBuffer;
  long unsigned     nbuf;       ///< Size of bytebuffer
  MediaSubsession&  fSubsession;
  char*             fStreamId;
  FrameFilter&      framefilter;
  Frame             setupframe; ///< This frame is used to send subsession information
  Frame             frame;      ///< Data is being copied into this frame
  int               subsession_index;

public: // getters & setters
  uint8_t* getReceiveBuffer() {return fReceiveBuffer;}
  
public:
  bool on;
};



