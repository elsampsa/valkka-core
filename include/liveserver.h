#ifndef SERVER_HEADER_GUARD 
#define SERVER_HEADER_GUARD

/*
 * liveserver.h : Live555 interface for server side: streaming to udp sockets directly or by using an on-demand rtsp server
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
 *  @file    liveserver.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.5.2 
 *  
 *  @brief   Live555 interface for server side: streaming to udp sockets directly or by using an on-demand rtsp server
 */ 


// TODO!  There is no way to close these suckers with Medium::close().  Then BasicUsageEnvironment can't be reclaimed at we get a (minor) memory leak upon exit.

#include "livedep.h"
#include "framefifo.h"
#include "frame.h"
#include "logging.h"


/** Implements a FramedSource for sending frames.  See \ref live_streaming_page
 * 
 * @ingroup live_tag
 */
class BufferSource: public FramedSource {

public:
  /** Default constructor
   * 
   * @param env    Identifies the live555 event loop
   * @param fifo   See BufferSource::fifo
   * 
   */
  BufferSource(UsageEnvironment &env, FrameFifo &fifo, Boolean &canary, unsigned preferredFrameSize =0, unsigned playTimePerFrame =0, unsigned offset=0);
  virtual ~BufferSource();
  
private:
  virtual void doGetNextFrame();  ///< All the fun happens here
  
private:
  FrameFifo &fifo;                ///< Frames are being read from here.  This reference leads all the way down to LiveThread::fifo
  Boolean   &canary;              ///< If this instance of BufferSource get's annihilated, kill the canary (set it to false)
  unsigned  fPreferredFrameSize;
  unsigned  fPlayTimePerFrame;
  unsigned  offset;
  
public:
  std::deque<BasicFrame*> internal_fifo; ///< Internal fifo BasicFrame, i.e. payload frames
  bool      active;             ///< If set, doGetNextFrame is currently re-scheduled
  
public:
  void handleFrame(Frame* f);   ///< Copies a Frame from BufferSource::fifo into BufferSource::internal_fifo.  Sets BufferSource::active.  Checks that FrameClass is FrameClass::basic
  
};


/** An outbound Stream
 * 
 * In the live555 API, there are filterchains as well.  These end to a sink, while the sink queries frames from the source.
 * 
 * Here the source is Stream::buffer_source (BufferSource) and the final sink is Stream::terminal.  Frames are fed with FrameFifo s into BufferSource.  If BufferSource has frames in it's BufferSource::internal_fifo, it will pass a frame down the live555 filterchain.
 * 
 * @ingroup live_tag
 */
class Stream {
  
public:
  /** Default constructor
   * 
   * @param env      Identifies the live555 event loop
   * @param fifo     See Stream::fifo
   * @param adr      Target address for sending the stream
   * @param portnum  Start port number for sending the stream
   * @param ttl      Packet time-to-live
   * 
   */
  Stream(UsageEnvironment &env, FrameFifo &fifo, const std::string adr, unsigned short int portnum, const unsigned char ttl=255);
  /** Default destructor */
  virtual ~Stream();
  
protected:
  UsageEnvironment  &env;   ///< Identifies the live555 event loop
  FrameFifo         &fifo;  ///< Frames are read from here.  This reference leads all the way down to LiveThread::fifo
  
  RTPSink           *sink;  ///< Live555 class: queries frames from terminal
  RTCPInstance      *rtcp;
  
  Groupsock  *rtpGroupsock;
  Groupsock  *rtcpGroupsock;
  unsigned char cname[101];
  
  BufferSource *buffer_source;   ///< Reserved in the child classes (depends on the payload type)
  FramedSource *terminal;        ///< The final sink in the live555 filterchain
  Boolean      source_alive;   ///< A canary variable that tells us if live555 event loop has closed the buffer_source
  
public:
  void handleFrame(Frame *f);
  void startPlaying();
  static void afterPlaying(void* cdata);
};



class ValkkaServerMediaSubsession: public OnDemandServerMediaSubsession {
  
protected: // we're a virtual base class
  ValkkaServerMediaSubsession(UsageEnvironment &env, FrameFifo &fifo, Boolean reuseFirstSource);
  virtual ~ValkkaServerMediaSubsession();

protected:
  // char const* fFileName;
  // u_int64_t fFileSize; // if known
  BufferSource *buffer_source;  ///< Reserved in the child classes (depends on the payload type)
  FrameFifo    &fifo;
  Boolean      source_alive;   ///< A canary variable that tells us if live555 event loop has closed the buffer_source
  
protected:
  virtual void setDoneFlag() =0; ///< call before removing this from the server .. informs the extra inner event loop (if any)
  
public:
  void handleFrame(Frame *f);   ///< Puts a frame into the buffer_source
};



class H264ServerMediaSubsession: public ValkkaServerMediaSubsession {
  
public:
  static H264ServerMediaSubsession* createNew(UsageEnvironment& env, FrameFifo &fifo, Boolean reuseFirstSource);
  // Used to implement "getAuxSDPLine()":
  void checkForAuxSDPLine1();
  void afterPlayingDummy1();

protected:
  static void afterPlayingDummy(void* clientData);
  static void checkForAuxSDPLine(void* clientData);
  
protected:
  H264ServerMediaSubsession(UsageEnvironment& env, FrameFifo &fifo, Boolean reuseFirstSource); // called only by createNew();
  virtual ~H264ServerMediaSubsession();
  void setDoneFlag() { fDoneFlag = ~0; }

protected: // redefined virtual functions
  virtual char const* getAuxSDPLine(RTPSink* rtpSink, FramedSource* inputSource);
  virtual FramedSource* createNewStreamSource(unsigned clientSessionId, unsigned& estBitrate); // returns BufferSource, framed with correct type
  virtual RTPSink* createNewRTPSink(Groupsock* rtpGroupsock, unsigned char rtpPayloadTypeIfDynamic, FramedSource* inputSource); // returns H264RTPSink

private:
  char* fAuxSDPLine;
  char fDoneFlag; // used when setting up "fAuxSDPLine"
  RTPSink* fDummyRTPSink; // ditto
};



class H264Stream : public Stream {
 
public:
  H264Stream(UsageEnvironment &env, FrameFifo &fifo, const std::string adr, unsigned short int portnum, const unsigned char ttl=255);
  ~H264Stream();
    
};

#endif

