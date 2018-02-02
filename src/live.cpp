/*
 * live.cpp : Interface to live555
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
 *  @file    live.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.0 
 *  
 *  @brief Interface to live555
 *
 *  @section DESCRIPTION
 *  
 *  Yes, the description
 *
 */
 
#include "live.h"
#include "sizes.h"
#include "tools.h"
#include "logging.h"

#define SEND_PARAMETER_SETS // keep this always defined!


BufferSource::BufferSource(UsageEnvironment& env, FrameFifo &fifo, unsigned preferredFrameSize, unsigned playTimePerFrame, unsigned offset) : FramedSource(env), fifo(fifo), fPreferredFrameSize(preferredFrameSize), fPlayTimePerFrame(playTimePerFrame), offset(offset), active(true) {
}

BufferSource::~BufferSource() {
  for(auto it=internal_fifo.begin(); it!=internal_fifo.end(); ++it) {
    fifo.recycle(*it);
  }
}


void BufferSource::handleFrame(Frame* f) {
  internal_fifo.push_front(f);
  if (!active) {
    doGetNextFrame();
  }
}


void BufferSource::doGetNextFrame() {
  /* // testing ..
  std::cout << "doGetNextFrame!\n";
  return;
  */
  
  /*
  http://lists.live555.com/pipermail/live-devel/2014-October/018768.html
  
  control flow:
  http://www.live555.com/liveMedia/faq.html#control-flow
  
  use of discrete framer:
  http://lists.live555.com/pipermail/live-devel/2014-September/018686.html
  http://lists.live555.com/pipermail/live-devel/2011-November/014019.html
  .. Ross getting annoyed:
  http://lists.live555.com/pipermail/live-devel/2016-January/019856.html
  
  this::doGetNextFrame => FramedSource::afterGetting(this) [resets fIsCurrentlyAwaitingData] => calls "AfterGettingFunc" callback (defined god-knows-where) => .. end up calling 
  FramedSource::getNextFrame [this tilts if fIsCurrentlyAwaitingData is set]
  http://lists.live555.com/pipermail/live-devel/2005-August/002995.html
  
  */
  
  /* // insight from "FramedSource.hh":
  // The following variables are typically accessed/set by doGetNextFrame()
  unsigned char* fTo; // in
  unsigned fMaxSize; // in
  
  unsigned fFrameSize; // out
  unsigned fNumTruncatedBytes; // out
  struct timeval fPresentationTime; // out
  unsigned fDurationInMicroseconds; // out
  */
  Frame* f;
  
  if (internal_fifo.empty()) {
    active=false; // this method is not re-scheduled anymore .. must be called again.
    return;
  }
  active=true; // this will be re-scheduled

  f=internal_fifo.back(); // take the last element
  
  fFrameSize         =(unsigned)f->payload.size()-offset;
  
  std::cout << "BufferSource : doGetNextFrame : fMaxSize     " << fMaxSize << std::endl;
  std::cout << "BufferSource : doGetNextFrame : payload size " << f->payload.size() << std::endl;
  std::cout << "BufferSource : doGetNextFrame : fFrameSize   " << fFrameSize << std::endl;
  
  // memcpy(fTo, f->payload.data(), f->payload.size());
  memcpy(fTo, f->payload.data()+offset, std::min(fFrameSize,fMaxSize));
  fNumTruncatedBytes =std::min((unsigned)0,fFrameSize-fMaxSize);
  // fMaxSize  =f->payload.size();
  // fNumTruncatedBytes=0;
  
  fPresentationTime=msToTimeval(f->mstimestamp); // timestamp to time struct
  
  // fDurationInMicroseconds = 0;
  // fPresentationTime.tv_sec   =(fMstimestamp/1000); // secs
  // fPresentationTime.tv_usec  =(fMstimestamp-fPresentationTime.tv_sec*1000)*1000; // microsecs
  // std::cout << "call_afterGetting: " << fPresentationTime.tv_sec << "." << fPresentationTime.tv_usec << " " << fFrameSize << "\n";
  std::cout << "calling afterGetting\n";

  fifo.recycle(f); // return the frame to the main live555 incoming fifo
  internal_fifo.pop_back();
  
  if (internal_fifo.empty()) {
    fDurationInMicroseconds = 0; // return immediately here and brake the re-scheduling
  }
  else { // approximate when this will be called again
    fDurationInMicroseconds = 0;
  }
  
  FramedSource::afterGetting(this);
}


Stream::Stream(UsageEnvironment &env, FrameFifo &fifo, const std::string adr, unsigned short int portnum, const unsigned char ttl) : env(env), fifo(fifo) {
  /*
  UsageEnvironment& env;
  RTPSink*          sink;
  RTCPInstance*     rtcp;
  FramedSource*     source;
  Groupsock* rtpGroupsock;
  Groupsock* rtcpGroupsock;
  unsigned char cname[101];

  BufferSource* buffer_source;
  */
  
  // Create 'groupsocks' for RTP and RTCP:
  struct in_addr destinationAddress;
  //destinationAddress.s_addr = chooseRandomIPv4SSMAddress(*env);
  destinationAddress.s_addr=our_inet_addr(adr.c_str());

  Port rtpPort(portnum);
  Port rtcpPort(portnum+1);

  // Groupsock rtpGroupsock(this->env, destinationAddress, rtpPort, ttl);
  this->rtpGroupsock  =new Groupsock(this->env, destinationAddress, rtpPort, ttl);
  // this->rtpGroupsock->multicastSendOnly();
  
  int fd=this->rtpGroupsock->socketNum();
  // increaseSendBufferTo(this->env,fd,this->nsocket); // TODO
  
  this->rtcpGroupsock =new Groupsock(this->env, destinationAddress, rtcpPort, ttl);
  // this->rtcpGroupsock->multicastSendOnly();
  // rtpGroupsock.multicastSendOnly(); // we're a SSM source
  // Groupsock rtcpGroupsock(*env, destinationAddress, rtcpPort, ttl);
  // rtcpGroupsock.multicastSendOnly(); // we're a SSM source

  // Create a 'H264 Video RTP' sink from the RTP 'groupsock':
  // OutPacketBuffer::maxSize = 100000;
  
  unsigned char CNAME[101];
  gethostname((char*)CNAME, 100);
  CNAME[100] = '\0'; // just in case ..
  memcpy(this->cname,CNAME,101);
}


void Stream::handleFrame(Frame* f) {
  buffer_source->handleFrame(f); // buffer source recycles the frame when ready
}

void Stream::startPlaying() {
  sink->startPlaying(*(terminal), this->afterPlaying, this);
}

void Stream::afterPlaying(void *cdata) {
  Stream* stream=(Stream*)cdata;
  
  stream->sink->stopPlaying();
  // Medium::close(stream->buffer_source);
}


Stream::~Stream() {
  Medium::close(buffer_source);
  delete rtpGroupsock;
  delete rtcpGroupsock;
  delete buffer_source;
}
  

  
H264Stream::H264Stream(UsageEnvironment &env, FrameFifo &fifo, const std::string adr, unsigned short int portnum, const unsigned char ttl) : Stream(env,fifo,adr,portnum,ttl) {
  
  buffer_source  =new BufferSource(env, fifo, 0, 0, 4); // nalstamp offset: 4
  
  // OutPacketBuffer::maxSize = this->npacket; // TODO
  // http://lists.live555.com/pipermail/live-devel/2013-April/016816.html
  
  
  sink = H264VideoRTPSink::createNew(env,rtpGroupsock, 96);
  // this->rtcp      = RTCPInstance::createNew(this->env, this->rtcpGroupsock, 500,  this->cname, sink, NULL, True); // saturates the event loop!
  terminal       =H264VideoStreamDiscreteFramer::createNew(env, buffer_source);
}


H264Stream::~H264Stream() {
  // delete sink;
  // delete terminal;
  Medium::close(sink);
  Medium::close(terminal);
}



// A function that outputs a string that identifies each stream (for debugging output).  Modify this if you wish:
UsageEnvironment& operator<<(UsageEnvironment& env, const RTSPClient& rtspClient) {
  return env << "[URL:\"" << rtspClient.url() << "\"]: ";
}

// A function that outputs a string that identifies each subsession (for debugging output).  Modify this if you wish:
UsageEnvironment& operator<<(UsageEnvironment& env, const MediaSubsession& subsession) {
  return env << subsession.mediumName() << "/" << subsession.codecName();
}

// A function that outputs a string that identifies each stream (for debugging output).  Modify this if you wish:
Logger& operator<<(Logger& logger, const RTSPClient& rtspClient) {
  return logger.log(logger.current_level) << "[URL:\"" << rtspClient.url() << "\"]: ";
}

// A function that outputs a string that identifies each subsession (for debugging output).  Modify this if you wish:
Logger& operator<<(Logger& logger, const MediaSubsession& subsession) {
  return logger.log(logger.current_level) << subsession.mediumName() << "/" << subsession.codecName();
}


void usage(UsageEnvironment& env, char const* progName) {
  livelogger.log(LogLevel::normal) << "Usage: " << progName << " <rtsp-url-1> ... <rtsp-url-N>\n";
  livelogger.log(LogLevel::normal) << "\t(where each <rtsp-url-i> is a \"rtsp://\" URL)\n";
}

// By default, we request that the server stream its data using RTP/UDP.
// If, instead, you want to request that the server stream via RTP-over-TCP, change the following to True:
#define REQUEST_STREAMING_OVER_TCP False

// Implementation of "ValkkaRTSPClient":

ValkkaRTSPClient* ValkkaRTSPClient::createNew(UsageEnvironment& env, const std::string rtspURL, FrameFilter& framefilter, LiveStatus* livestatus, int verbosityLevel, char const* applicationName, portNumBits tunnelOverHTTPPortNum) {
  return new ValkkaRTSPClient(env, rtspURL, framefilter, livestatus, verbosityLevel, applicationName, tunnelOverHTTPPortNum);
}

ValkkaRTSPClient::ValkkaRTSPClient(UsageEnvironment& env, const std::string rtspURL, FrameFilter& framefilter, LiveStatus* livestatus, int verbosityLevel, char const* applicationName, portNumBits tunnelOverHTTPPortNum) : RTSPClient(env, rtspURL.c_str(), verbosityLevel, applicationName, tunnelOverHTTPPortNum, -1), framefilter(framefilter), livestatus(livestatus)
{
}

ValkkaRTSPClient::~ValkkaRTSPClient() 
{
}


void ValkkaRTSPClient::continueAfterDESCRIBE(RTSPClient* rtspClient, int resultCode, char* resultString) {
  do {
    UsageEnvironment& env = rtspClient->envir(); // alias
    StreamClientState& scs = ((ValkkaRTSPClient*)rtspClient)->scs; // alias

    if (resultCode != 0) {
      livelogger.log(LogLevel::normal) << "ValkkaRTSPClient: " << *rtspClient << "Failed to get a SDP description: " << resultString << "\n";
      delete[] resultString;
      break;
    }

    char* const sdpDescription = resultString;
    // livelogger.log(LogLevel::normal) << *rtspClient << "Got a SDP description:\n" << sdpDescription << "\n";
    livelogger.log(LogLevel::debug) << "ValkkaRTSPClient: Got a SDP description:\n" << sdpDescription << "\n";

    // Create a media session object from this SDP description:
    scs.session = MediaSession::createNew(env, sdpDescription);
    delete[] sdpDescription; // because we don't need it anymore
    if (scs.session == NULL) {
      // livelogger.log(LogLevel::normal) << *rtspClient << "Failed to create a MediaSession object from the SDP description: " << env.getResultMsg() << "\n";
      livelogger.log(LogLevel::normal) << "ValkkaRTSPClient: Failed to create a MediaSession object from the SDP description: " << env.getResultMsg() << "\n";
      break;
    } else if (!scs.session->hasSubsessions()) {
      // livelogger.log(LogLevel::normal) << *rtspClient << "This session has no media subsessions (i.e., no \"m=\" lines)\n";
      livelogger.log(LogLevel::normal) << "ValkkaRTSPClient: This session has no media subsessions (i.e., no \"m=\" lines)\n";
      break;
    }

    // Then, create and set up our data source objects for the session.  We do this by iterating over the session's 'subsessions',
    // calling "MediaSubsession::initiate()", and then sending a RTSP "SETUP" command, on each one.
    // (Each 'subsession' will have its own data source.)
    scs.iter = new MediaSubsessionIterator(*scs.session);
    setupNextSubsession(rtspClient);
    return;
  } while (0);

  // An unrecoverable error occurred with this stream.
  shutdownStream(rtspClient);
}

void ValkkaRTSPClient::setupNextSubsession(RTSPClient* rtspClient) {
  UsageEnvironment& env = rtspClient->envir(); // alias
  StreamClientState& scs = ((ValkkaRTSPClient*)rtspClient)->scs; // alias
  bool ok_subsession_type = false;
  
  scs.subsession = scs.iter->next();
  scs.subsession_index++;
  
  // CAM_EXCEPTION : UNV-1 == MANUFACTURER: UNV MODEL: IPC312SR-VPF28 
  // CAM_EXCEPTION : UNV-1 : Some UNV cameras go crazy if you try to issue SETUP on the "metadata" subsession which they themselves provide
  // CAM_EXCEPTION : UNV-1 : I get "failed to setup subsession", which is ok, but when "PLAY" command is issued (not on the metadata subsession, but just to normal session) 
  // CAM_EXCEPTION : UNV-1 : we get "connection reset by peer" at continueAfterPlay
  // ok_subsession_type = (strcmp(scs.subsession->mediumName(),"video")==0 or strcmp(scs.subsession->mediumName(),"audio")==0); // CAM_EXCEPTION : UNV-1
  ok_subsession_type = true; // TODO: debug this..! why it crashes sometimes?
  
  if (scs.subsession != NULL and ok_subsession_type) {
    if (!scs.subsession->initiate()) {
      livelogger.log(LogLevel::normal) << "ValkkaRTSPClient: " << *rtspClient << "Failed to initiate the \"" << *scs.subsession << "\" subsession: " << env.getResultMsg() << "\n";
      setupNextSubsession(rtspClient); // give up on this subsession; go to the next one
    } else {
      livelogger.log(LogLevel::debug) << "ValkkaRTSPClient: " << *rtspClient << " Initiated the \"" << *scs.subsession << "\" subsession (";
      if (scs.subsession->rtcpIsMuxed()) {
        livelogger.log(LogLevel::debug) << "client port " << scs.subsession->clientPortNum();
      } else {
        livelogger.log(LogLevel::debug) << "client ports " << scs.subsession->clientPortNum() << "-" << scs.subsession->clientPortNum()+1;
      }
      livelogger.log(LogLevel::debug) << ")\n";

      // Continue setting up this subsession, by sending a RTSP "SETUP" command:
      rtspClient->sendSetupCommand(*scs.subsession, continueAfterSETUP, False, REQUEST_STREAMING_OVER_TCP);
    }
    return;
  }

  // We've finished setting up all of the subsessions.  Now, send a RTSP "PLAY" command to start the streaming:
  if (scs.session->absStartTime() != NULL) {
    // Special case: The stream is indexed by 'absolute' time, so send an appropriate "PLAY" command:
    rtspClient->sendPlayCommand(*scs.session, continueAfterPLAY, scs.session->absStartTime(), scs.session->absEndTime());
  } else {
    scs.duration = scs.session->playEndTime() - scs.session->playStartTime();
    rtspClient->sendPlayCommand(*scs.session, continueAfterPLAY);
  }
}

void ValkkaRTSPClient::continueAfterSETUP(RTSPClient* rtspClient, int resultCode, char* resultString) {
  do {
    UsageEnvironment& env    = rtspClient->envir(); // alias
    StreamClientState& scs   = ((ValkkaRTSPClient*)rtspClient)->scs; // alias
    FrameFilter& framefilter = ((ValkkaRTSPClient*)rtspClient)->framefilter;

    if (resultCode != 0) {
      livelogger.log(LogLevel::normal) << "ValkkaRTSPClient: " << *rtspClient << "Failed to set up the \"" << *scs.subsession << "\" subsession: " << resultString << "\n";
      break;
    }

    // livelogger.log(LogLevel::normal) << *rtspClient << "Set up the \"" << *scs.subsession << "\" subsession (";
    livelogger.log(LogLevel::debug) << "ValkkaRTSPClient: " << *rtspClient << "Set up the \"" << *scs.subsession << "\" subsession (";
    if (scs.subsession->rtcpIsMuxed()) {
      // livelogger.log(LogLevel::normal) << "client port " << scs.subsession->clientPortNum();
      livelogger.log(LogLevel::debug) << "client port " << scs.subsession->clientPortNum();
    } else {
      // livelogger.log(LogLevel::normal) << "client ports " << scs.subsession->clientPortNum() << "-" << scs.subsession->clientPortNum()+1;
      livelogger.log(LogLevel::debug) << "client ports " << scs.subsession->clientPortNum() << "-" << scs.subsession->clientPortNum()+1;
    }
    // livelogger.log(LogLevel::normal) << ")\n";
    livelogger.log(LogLevel::debug) << ")\n";

    // Having successfully setup the subsession, create a data sink for it, and call "startPlaying()" on it.
    // (This will prepare the data sink to receive data; the actual flow of data from the client won't start happening until later,
    // after we've sent a RTSP "PLAY" command.)

    scs.subsession->sink = FrameSink::createNew(env, *scs.subsession, framefilter, scs.subsession_index, rtspClient->url());
      // perhaps use your own custom "MediaSink" subclass instead
    if (scs.subsession->sink == NULL) {
      livelogger.log(LogLevel::normal) << "ValkkaRTSPClient: " << *rtspClient << "Failed to create a data sink for the \"" << *scs.subsession
	  << "\" subsession: " << env.getResultMsg() << "\n";
      break;
    }

    livelogger.log(LogLevel::debug) << "ValkkaRTSPClient: " << *rtspClient << "Created a data sink for the \"" << *scs.subsession << "\" subsession\n";
    scs.subsession->miscPtr = rtspClient; // a hack to let subsession handler functions get the "RTSPClient" from the subsession 
    scs.subsession->sink->startPlaying(*(scs.subsession->readSource()),
				       subsessionAfterPlaying, scs.subsession);
    // Also set a handler to be called if a RTCP "BYE" arrives for this subsession:
    if (scs.subsession->rtcpInstance() != NULL) {
      scs.subsession->rtcpInstance()->setByeHandler(subsessionByeHandler, scs.subsession);
    }
  } while (0);
  delete[] resultString;

  // Set up the next subsession, if any:
  setupNextSubsession(rtspClient);
}


void ValkkaRTSPClient::continueAfterPLAY(RTSPClient* rtspClient, int resultCode, char* resultString) {
  Boolean success = False;
  LiveStatus* livestatus = ((ValkkaRTSPClient*)rtspClient)->livestatus; // alias

  do {
    UsageEnvironment& env = rtspClient->envir(); // alias
    StreamClientState& scs = ((ValkkaRTSPClient*)rtspClient)->scs; // alias

    if (resultCode != 0) {
      livelogger.log(LogLevel::normal) << "ValkkaRTSPClient: " << *rtspClient << " Failed to start playing session: " << resultString << "\n";
      break;
    }

    // Set a timer to be handled at the end of the stream's expected duration (if the stream does not already signal its end
    // using a RTCP "BYE").  This is optional.  If, instead, you want to keep the stream active - e.g., so you can later
    // 'seek' back within it and do another RTSP "PLAY" - then you can omit this code.
    // (Alternatively, if you don't want to receive the entire stream, you could set this timer for some shorter value.)
    if (scs.duration > 0) {
      unsigned const delaySlop = 2; // number of seconds extra to delay, after the stream's expected duration.  (This is optional.)
      scs.duration += delaySlop;
      unsigned uSecsToDelay = (unsigned)(scs.duration*1000000);
      scs.streamTimerTask = env.taskScheduler().scheduleDelayedTask(uSecsToDelay, (TaskFunc*)streamTimerHandler, rtspClient);
    }

    livelogger.log(LogLevel::debug) << "ValkkaRTSPClient: " << *rtspClient << "Started playing session";
    if (scs.duration > 0) {
      livelogger.log(LogLevel::debug) << " (for up to " << scs.duration << " seconds)";
    }
    livelogger.log(LogLevel::debug) << "...\n";

    success = True;
  } while (0);
  delete[] resultString;

  if (!success) {
    // An unrecoverable error occurred with this stream.
    shutdownStream(rtspClient);
  }
  else {
   *livestatus=LiveStatus::alive;
  }
}


// Implementation of the other event handlers:

void ValkkaRTSPClient::subsessionAfterPlaying(void* clientData) {
  MediaSubsession* subsession = (MediaSubsession*)clientData;
  RTSPClient* rtspClient = (RTSPClient*)(subsession->miscPtr);

  // Begin by closing this subsession's stream:
  Medium::close(subsession->sink);
  subsession->sink = NULL;

  // Next, check whether *all* subsessions' streams have now been closed:
  MediaSession& session = subsession->parentSession();
  MediaSubsessionIterator iter(session);
  while ((subsession = iter.next()) != NULL) {
    if (subsession->sink != NULL) return; // this subsession is still active
  }

  // All subsessions' streams have now been closed, so shutdown the client:
  shutdownStream(rtspClient);
}


void ValkkaRTSPClient::subsessionByeHandler(void* clientData) {
  MediaSubsession* subsession = (MediaSubsession*)clientData;
  RTSPClient* rtspClient = (RTSPClient*)subsession->miscPtr;
  UsageEnvironment& env = rtspClient->envir(); // alias

  livelogger.log(LogLevel::debug) << "ValkkaRTSPClient: " << *rtspClient << "Received RTCP \"BYE\" on \"" << *subsession << "\" subsession\n";

  // Now act as if the subsession had closed:
  subsessionAfterPlaying(subsession);
}

void ValkkaRTSPClient::streamTimerHandler(void* clientData) {
  ValkkaRTSPClient* rtspClient = (ValkkaRTSPClient*)clientData;
  StreamClientState& scs = rtspClient->scs; // alias

  scs.streamTimerTask = NULL;

  // Shut down the stream:
  shutdownStream(rtspClient);
}

void ValkkaRTSPClient::shutdownStream(RTSPClient* rtspClient, int exitCode) {
  UsageEnvironment& env  =rtspClient->envir(); // alias
  StreamClientState& scs =((ValkkaRTSPClient*)rtspClient)->scs; // alias
  LiveStatus* livestatus = ((ValkkaRTSPClient*)rtspClient)->livestatus; // alias
  
  livelogger.log(LogLevel::debug) << "ValkkaRTSPClient: shutdownStream :" <<std::endl;
  
  // First, check whether any subsessions have still to be closed:
  if (scs.session != NULL) { 
    Boolean someSubsessionsWereActive = False;
    MediaSubsessionIterator iter(*scs.session);
    MediaSubsession* subsession;

    while ((subsession = iter.next()) != NULL) {
      if (subsession->sink != NULL) {
        livelogger.log(LogLevel::debug) << "ValkkaRTSPClient: shutdownStream : closing subsession" <<std::endl;
	Medium::close(subsession->sink);
        livelogger.log(LogLevel::debug) << "ValkkaRTSPClient: shutdownStream : closed subsession" <<std::endl;
	subsession->sink = NULL;

	if (subsession->rtcpInstance() != NULL) {
	  subsession->rtcpInstance()->setByeHandler(NULL, NULL); // in case the server sends a RTCP "BYE" while handling "TEARDOWN"
	}

	someSubsessionsWereActive = True;
      }
    }

    if (someSubsessionsWereActive) {
      // Send a RTSP "TEARDOWN" command, to tell the server to shutdown the stream.
      // Don't bother handling the response to the "TEARDOWN".
      livelogger.log(LogLevel::debug) << "ValkkaRTSPClient: shutdownStream : sending teardown" <<std::endl;
      rtspClient->sendTeardownCommand(*scs.session, NULL);
    }
  }

  livelogger.log(LogLevel::debug) << *rtspClient << "Closing the stream.\n";
  *livestatus=LiveStatus::closed;
  Medium::close(rtspClient);
  // Note that this will also cause this stream's "StreamClientState" structure to get reclaimed.
  // Uh-oh: how do we tell the event loop that this particular client does not exist anymore..?
  
  /*
  if (--rtspClientCount == 0) {
    // The final stream has ended, so exit the application now.
    // (Of course, if you're embedding this code into your own application, you might want to comment this out,
    // and replace it with "eventLoopWatchVariable = 1;", so that we leave the LIVE555 event loop, and continue running "main()".)
    exit(exitCode);
  }
  */
}

// Implementation of "StreamClientState":

StreamClientState::StreamClientState() : iter(NULL), session(NULL), subsession(NULL), streamTimerTask(NULL), duration(0.0), subsession_index(-1)
{
}

StreamClientState::~StreamClientState() {
  delete iter;
  if (session != NULL) {
    // We also need to delete "session", and unschedule "streamTimerTask" (if set)
    UsageEnvironment& env = session->envir(); // alias

    env.taskScheduler().unscheduleDelayedTask(streamTimerTask);
    Medium::close(session);
  }
}


// Implementation of "FrameSink":

// Even though we're not going to be doing anything with the incoming data, we still need to receive it.
// Define the size of the buffer that we'll use:
#define DUMMY_SINK_RECEIVE_BUFFER_SIZE 100000

FrameSink* FrameSink::createNew(UsageEnvironment& env, MediaSubsession& subsession, FrameFilter& framefilter, int subsession_index, char const* streamId) {
  return new FrameSink(env, subsession, framefilter, subsession_index, streamId);
}

FrameSink::FrameSink(UsageEnvironment& env, MediaSubsession& subsession, FrameFilter& framefilter, int subsession_index, char const* streamId) : MediaSink(env), fSubsession(subsession), framefilter(framefilter), subsession_index(subsession_index), on(true), nbuf(0) {
  fStreamId = strDup(streamId);
  // fReceiveBuffer = new u_int8_t[DUMMY_SINK_RECEIVE_BUFFER_SIZE];
  
  const char* codec_name=subsession.codecName();
  
  livelogger.log(LogLevel::debug) << "FrameSink: constructor: codec_name ="<< codec_name << ", subsession_index ="<<subsession_index <<std::endl;
  
  // https://ffmpeg.org/doxygen/3.0/avcodec_8h_source.html
  if      (strcmp(codec_name,"H264")==0) { // NEW_CODEC_DEV // when adding new codecs, make changes here
    livelogger.log(LogLevel::debug) << "FrameSink: init H264 Frame"<<std::endl;
    // prepare payload frame
    frame.frametype            =FrameType::h264; // this is h264
    frame.subsession_index     =subsession_index;
    frame.h264_pars.slice_type =0;
    // frame.payload.resize(DEFAULT_PAYLOAD_SIZE_H264+4); // leave space for prepending 0001
    setReceiveBuffer(DEFAULT_PAYLOAD_SIZE_H264); // sets nbuf
    // prepare setup frame
    setupframe.frametype=FrameType::setup; // this is a setup frame
    setupframe.setup_pars.frametype=FrameType::h264; // what frame types are to be expected from this stream
    // setupframe.setup_pars.subsession_index=subsession_index;
    setupframe.subsession_index=subsession_index;
    setupframe.setMsTimestamp(getCurrentMsTimestamp());
    // send setup frame
    framefilter.run(&setupframe);
  } 
  else if (strcmp(codec_name,"PCMU")==0) {
    livelogger.log(LogLevel::debug) << "FrameSink: init PCMU Frame"<<std::endl;
    // prepare payload frame
    frame.frametype       =FrameType::pcmu;
    frame.subsession_index=subsession_index; // pcmu
    // frame.payload.resize(DEFAULT_PAYLOAD_SIZE_PCMU);
    setReceiveBuffer(DEFAULT_PAYLOAD_SIZE_PCMU); // sets nbuf
    // prepare setup frame
    setupframe.frametype=FrameType::setup; // this is a setup frame
    setupframe.setup_pars.frametype=FrameType::pcmu; // what frame types are to be expected from this stream
    // setupframe.setup_pars.subsession_index=subsession_index;
    setupframe.subsession_index=subsession_index;
    setupframe.setMsTimestamp(getCurrentMsTimestamp());
    // send setup frame
    framefilter.run(&setupframe);
  }
  else {
    livelogger.log(LogLevel::debug) << "FrameSink: WARNING: unknown codec "<<codec_name<<std::endl;
    frame.frametype=FrameType::none;
    setReceiveBuffer(DEFAULT_PAYLOAD_SIZE); // sets nbuf
  }
  
#ifdef SEND_PARAMETER_SETS
  sendParameterSets();
#endif
  
  livelogger.log(LogLevel::debug) << "FrameSink: constructor: internal_frame= "<< frame <<std::endl;
  
}

FrameSink::~FrameSink() {
  livelogger.log(LogLevel::crazy) << "FrameSink: destructor :"<<std::endl;
  // delete[] fReceiveBuffer;
  delete[] fStreamId;
  livelogger.log(LogLevel::crazy) << "FrameSink: destructor : bye!"<<std::endl;
}

void FrameSink::afterGettingFrame(void* clientData, unsigned frameSize, unsigned numTruncatedBytes, struct timeval presentationTime, unsigned durationInMicroseconds) {
  FrameSink* sink = (FrameSink*)clientData;
  sink->afterGettingFrame(frameSize, numTruncatedBytes, presentationTime, durationInMicroseconds);
}

// If you don't want to see debugging output for each received frame, then comment out the following line:
// #define DEBUG_PRINT_EACH_RECEIVED_FRAME 1

void FrameSink::afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes, struct timeval presentationTime, unsigned /*durationInMicroseconds*/) {
  // We've just received a frame of data.  (Optionally) print out information about it:
#ifdef DEBUG_PRINT_EACH_RECEIVED_FRAME
  if (fStreamId != NULL) envir() << "Stream \"" << fStreamId << "\"; ";
  envir() << fSubsession.mediumName() << "/" << fSubsession.codecName() << ":\tReceived " << frameSize << " bytes";
  if (numTruncatedBytes > 0) envir() << " (with " << numTruncatedBytes << " bytes truncated)";
  char uSecsStr[6+1]; // used to output the 'microseconds' part of the presentation time
  sprintf(uSecsStr, "%06u", (unsigned)presentationTime.tv_usec);
  envir() << ".\tPresentation time: " << (int)presentationTime.tv_sec << "." << uSecsStr;
  if (fSubsession.rtpSource() != NULL && !fSubsession.rtpSource()->hasBeenSynchronizedUsingRTCP()) {
    envir() << " !"; // mark the debugging output to indicate that this presentation time is not RTCP-synchronized
  }
#ifdef DEBUG_PRINT_NPT
  envir() << "\tNPT: " << fSubsession.getNormalPlayTime(presentationTime);
#endif
  envir() << "\n";
#endif
  
  unsigned target_size=frameSize+numTruncatedBytes;
  // mstimestamp=presentationTime.tv_sec*1000+presentationTime.tv_usec/1000;
  // std::cout << "afterGettingFrame: mstimestamp=" << mstimestamp <<std::endl;
  frame.setMsTimestamp(presentationTime.tv_sec*1000+presentationTime.tv_usec/1000);
  frame.fillPars();
  /*
  std::cout << ">>nbuf="<<nbuf<<std::endl;
  int cc=0;
  for(auto it=frame.payload.begin(); it!=frame.payload.end(); ++it) {
    std::cout << "a>>" << int(*it) << std::endl;
    cc++;
    if (cc>30) { break; }
  }
  
  std::cout << ">>resizing to " << checkBufferSize(frameSize+numTruncatedBytes) << std::endl;
  
  cc=0;
  for(auto it=frame.payload.begin(); it!=frame.payload.end(); ++it) {
    std::cout << "b>>" << int(*it) << std::endl;
    cc++;
    if (cc>30) { break; }
  }
  */
  frame.payload.resize(checkBufferSize(frameSize)); // set correct frame size .. now information about the packet length goes into the filter chain
  
  framefilter.run(&frame); // starts the frame filter chain
  
  if (numTruncatedBytes>0) {// time to grow the buffer..
    livelogger.log(LogLevel::debug) << "FrameSink : growing reserved size to "<< target_size << " bytes" << std::endl;
    setReceiveBuffer(target_size);
  }
  
  frame.payload.resize(frame.payload.capacity()); // recovers maximum size .. must set maximum size before letting live555 to write into the memory area
  
  // Then continue, to request the next frame of data:
  if (on) {continuePlaying();}
}


Boolean FrameSink::continuePlaying() {
  if (fSource == NULL) return False; // sanity check (should not happen)
  // Request the next frame of data from our input source.  "afterGettingFrame()" will get called later, when it arrives:
  // fSource->getNextFrame(fReceiveBuffer, DUMMY_SINK_RECEIVE_BUFFER_SIZE, afterGettingFrame, this, onSourceClosure, this);
  fSource->getNextFrame(fReceiveBuffer, nbuf, afterGettingFrame, this, onSourceClosure, this);
  return True;
}


unsigned FrameSink::checkBufferSize(unsigned target_size) {// add something to the target_size, if needed (for H264, the nalstamp)
   if (frame.frametype==FrameType::h264) {
     target_size+=nalstamp.size();
   }
   // target_size+=8; // receive extra mem to avoid ffmpeg decoder over-read // nopes!  This can screw up the decoder completely!
   return target_size;
}


void FrameSink::setReceiveBuffer(unsigned target_size) {
  target_size=checkBufferSize(target_size); // correct buffer size to include nalstamp
  frame.payload.resize(target_size);
  if (frame.frametype==FrameType::h264) {
    fReceiveBuffer=frame.payload.data()+nalstamp.size(); // pointer for the beginning of the payload (after the nalstamp)
    nbuf=frame.payload.size()-nalstamp.size();           // size left for actual payload (without the prepending 0001)
    std::copy(nalstamp.begin(),nalstamp.end(),frame.payload.begin());
  }
  else {
    fReceiveBuffer=frame.payload.data();
    nbuf=frame.payload.size();
  }
  // std::cout << ">>re-setting receive buffer "<<nbuf<<std::endl;
}


void FrameSink::sendParameterSets() {
  SPropRecord* pars;
  unsigned i,num;
  struct timeval frametime;
  gettimeofday(&frametime, NULL);
  
  livelogger.log(LogLevel::crazy) << "Sending parameter sets!\n";
  pars=parseSPropParameterSets(fSubsession.fmtp_spropparametersets(),num);
  livelogger.log(LogLevel::crazy) << "Found " << num << " parameter sets\n";
  for(i=0;i<num;i++) {
    if (pars[i].sPropLength>0) {
      livelogger.log(LogLevel::crazy) << "Sending parameter set " << i << " " << pars[i].sPropLength << "\n";
      memcpy(fReceiveBuffer, pars[i].sPropBytes, pars[i].sPropLength);
      afterGettingFrame(pars[i].sPropLength, 0, frametime, 0);
    }
  }
}


