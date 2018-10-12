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
 *  @version 0.7.1 
 *  
 *  @brief Interface to live555
 *
 *  @section DESCRIPTION
 *  
 *  Yes, the description
 *
 */
 
#include "live.h"
#include "constant.h"
#include "tools.h"
#include "logging.h"

#define SEND_PARAMETER_SETS // keep this always defined

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

// Implementation of "ValkkaRTSPClient":

ValkkaRTSPClient* ValkkaRTSPClient::createNew(UsageEnvironment& env, const std::string rtspURL, FrameFilter& framefilter, LiveStatus* livestatus, int verbosityLevel, char const* applicationName, portNumBits tunnelOverHTTPPortNum) {
  return new ValkkaRTSPClient(env, rtspURL, framefilter, livestatus, verbosityLevel, applicationName, tunnelOverHTTPPortNum);
}

ValkkaRTSPClient::ValkkaRTSPClient(UsageEnvironment& env, const std::string rtspURL, FrameFilter& framefilter, LiveStatus* livestatus, int verbosityLevel, char const* applicationName, portNumBits tunnelOverHTTPPortNum) : RTSPClient(env, rtspURL.c_str(), verbosityLevel, applicationName, tunnelOverHTTPPortNum, -1), framefilter(framefilter), livestatus(livestatus), request_multicast(false), request_tcp(false), recv_buffer_size(0), reordering_time(0) {
}


ValkkaRTSPClient::~ValkkaRTSPClient() {
}


void ValkkaRTSPClient::continueAfterDESCRIBE(RTSPClient* rtspClient, int resultCode, char* resultString) {
  LiveStatus* livestatus = ((ValkkaRTSPClient*)rtspClient)->livestatus; // alias
  
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
  shutdownStream(rtspClient); // sets *livestatus=LiveStatus::closed;
}

void ValkkaRTSPClient::setupNextSubsession(RTSPClient* rtspClient) {
  // aliases:
  UsageEnvironment& env    = rtspClient->envir();
  StreamClientState& scs   = ((ValkkaRTSPClient*)rtspClient)->scs;
  ValkkaRTSPClient* client = (ValkkaRTSPClient*)rtspClient;
  LiveStatus* livestatus   = ((ValkkaRTSPClient*)rtspClient)->livestatus; // alias
  bool ok_subsession_type = false;
  
  scs.subsession = scs.iter->next();
  scs.subsession_index++;
  
  // CAM_EXCEPTION : UNV-1 == MANUFACTURER: UNV MODEL: IPC312SR-VPF28 
  // CAM_EXCEPTION : UNV-1 : Some UNV cameras go crazy if you try to issue SETUP on the "metadata" subsession which they themselves provide
  // CAM_EXCEPTION : UNV-1 : I get "failed to setup subsession", which is ok, but when "PLAY" command is issued (not on the metadata subsession, but just to normal session) 
  // CAM_EXCEPTION : UNV-1 : we get "connection reset by peer" at continueAfterPlay
  
  if (scs.subsession != NULL) { // has subsession
    
    livelogger.log(LogLevel::debug) << "ValkkaRTSPClient: handling subsession " << scs.subsession->mediumName() << std::endl;
    ok_subsession_type = (strcmp(scs.subsession->mediumName(),"video")==0 or strcmp(scs.subsession->mediumName(),"audio")==0); // CAM_EXCEPTION : UNV-1
    
    if (ok_subsession_type) { // a decent subsession
    
      if (!scs.subsession->initiate()) {
        livelogger.log(LogLevel::normal) << "ValkkaRTSPClient: " << *rtspClient << "Failed to initiate the \"" << *scs.subsession << "\" subsession: " << env.getResultMsg() << "\n";
        setupNextSubsession(rtspClient); // give up on this subsession; go to the next one
      } else { // subsession ok
        livelogger.log(LogLevel::debug) << "ValkkaRTSPClient: " << *rtspClient << " Initiated the \"" << *scs.subsession << "\" subsession (";
        if (scs.subsession->rtcpIsMuxed()) {
          livelogger.log(LogLevel::debug) << "client port " << scs.subsession->clientPortNum();
        } else {
          livelogger.log(LogLevel::debug) << "client ports " << scs.subsession->clientPortNum() << "-" << scs.subsession->clientPortNum()+1;
        }
        livelogger.log(LogLevel::debug) << ")\n";

        // adjust receive buffer size and reordering treshold time if requested
        if (scs.subsession->rtpSource() != NULL) {
          if (client->reordering_time>0) {
            scs.subsession->rtpSource()->setPacketReorderingThresholdTime(client->reordering_time);
            livelogger.log(LogLevel::normal) << "ValkkaRTSPClient: packet reordering time now " << client->reordering_time << " microseconds " << std::endl;
          }
          if (client->recv_buffer_size>0) {
            int socketNum = scs.subsession->rtpSource()->RTPgs()->socketNum();
            unsigned curBufferSize = getReceiveBufferSize(env, socketNum);
            unsigned newBufferSize = setReceiveBufferTo  (env, socketNum, client->recv_buffer_size);
            livelogger.log(LogLevel::normal) << "ValkkaRTSPClient: receiving socket size changed from " << curBufferSize << " to " << newBufferSize << std::endl;
          }
        }
        
        // Continue setting up this subsession, by sending a RTSP "SETUP" command:
        rtspClient->sendSetupCommand(*scs.subsession, continueAfterSETUP, False, client->request_tcp, client->request_multicast);
        
        /*
        unsigned RTSPClient::sendSetupCommand 	( 	MediaSubsession &  	subsession,
                  responseHandler *  	responseHandler,
                  Boolean  	streamOutgoing = False,
                  Boolean  	streamUsingTCP = False,
                  Boolean  	forceMulticastOnUnspecified = False,
                  Authenticator *  	authenticator = NULL 
          ) 	
        */
        
      } // subsession ok
    }
    else { // decent subsession
      livelogger.log(LogLevel::debug) << "ValkkaRTSPClient: discarded subsession " << scs.subsession->mediumName() << std::endl;
      setupNextSubsession(rtspClient); // give up on this subsession; go to the next one
    } // decent subsession
    return; // we have either called this routine again with another subsession or sent a setup command
  } // has subsession
    
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
  LiveStatus* livestatus = ((ValkkaRTSPClient*)rtspClient)->livestatus; // alias
  
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

    // scs.subsession->sink = FrameSink::createNew(env, *scs.subsession, framefilter, scs.subsession_index, rtspClient->url());
    scs.subsession->sink = FrameSink::createNew(env, scs, framefilter, rtspClient->url());
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
  LiveStatus* livestatus = ((ValkkaRTSPClient*)rtspClient)->livestatus; // alias

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

  livelogger.log(LogLevel::debug) << "ValkkaRTSPClient: " << *rtspClient << " closing the stream.\n";
  *livestatus=LiveStatus::closed;
  Medium::close(rtspClient);
  // Note that this will also cause this stream's "StreamClientState" structure to get reclaimed.
  // Uh-oh: how do we tell the event loop that this particular client does not exist anymore..?
  // .. before this RTSPClient deletes itself, we have modified the livestatus member .. that points to
  // a variable managed by the live thread
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

StreamClientState::StreamClientState() : iter(NULL), session(NULL), subsession(NULL), streamTimerTask(NULL), duration(0.0), subsession_index(-1), frame_flag(false)
{
}


void StreamClientState::close() {
  MediaSubsessionIterator iter2(*session);
  while ((subsession = iter2.next()) != NULL) {
    if (subsession->sink != NULL) {
      livelogger.log(LogLevel::debug) << "StreamClientState : closing subsession" <<std::endl;
      Medium::close(subsession->sink);
      livelogger.log(LogLevel::debug) << "StreamClientState : closed subsession" <<std::endl;
      subsession->sink = NULL;
    }
  }
}


StreamClientState::~StreamClientState() {
  delete iter;
  // return;
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

/*
FrameSink* FrameSink::createNew(UsageEnvironment& env, MediaSubsession& subsession, FrameFilter& framefilter, int subsession_index, char const* streamId) {
  return new FrameSink(env, subsession, framefilter, subsession_index, streamId);
}

FrameSink::FrameSink(UsageEnvironment& env, MediaSubsession& subsession, FrameFilter& framefilter, int subsession_index, char const* streamId) : MediaSink(env), fSubsession(subsession), framefilter(framefilter), subsession_index(subsession_index), on(true), nbuf(0) 
*/

FrameSink* FrameSink::createNew(UsageEnvironment& env, StreamClientState& scs, FrameFilter& framefilter, char const* streamId) {
  return new FrameSink(env, scs, framefilter, streamId);
}

FrameSink::FrameSink(UsageEnvironment& env, StreamClientState& scs, FrameFilter& framefilter, char const* streamId) : MediaSink(env), scs(scs), framefilter(framefilter), on(true), nbuf(0), fSubsession(*(scs.subsession))

{
  // some aliases:
  // MediaSubsession &subsession = *(scs.subsession);
  int subsession_index        = scs.subsession_index;
  
  fStreamId = strDup(streamId);
  // fReceiveBuffer = new u_int8_t[DUMMY_SINK_RECEIVE_BUFFER_SIZE];
  
  const char* codec_name=fSubsession.codecName();
  
  livelogger.log(LogLevel::debug) << "FrameSink: constructor: codec_name ="<< codec_name << ", subsession_index ="<<subsession_index <<std::endl;
  
  // https://ffmpeg.org/doxygen/3.0/avcodec_8h_source.html
  if      (strcmp(codec_name,"H264")==0) { // NEW_CODEC_DEV // when adding new codecs, make changes here
    livelogger.log(LogLevel::debug) << "FrameSink: init H264 Frame"<<std::endl;
    // prepare payload frame
    basicframe.media_type           =AVMEDIA_TYPE_VIDEO;
    basicframe.codec_id             =AV_CODEC_ID_H264;
    basicframe.subsession_index     =subsession_index;
    // prepare setup frame
    setupframe.media_type           =AVMEDIA_TYPE_VIDEO;
    setupframe.codec_id             =AV_CODEC_ID_H264;   // what frame types are to be expected from this stream
    setupframe.subsession_index     =subsession_index;
    setupframe.mstimestamp          =getCurrentMsTimestamp();
    // send setup frame
    framefilter.run(&setupframe);
    setReceiveBuffer(DEFAULT_PAYLOAD_SIZE_H264); // sets nbuf
  } 
  else if (strcmp(codec_name,"PCMU")==0) {
    livelogger.log(LogLevel::debug) << "FrameSink: init PCMU Frame"<<std::endl;
    // prepare payload frame
    basicframe.media_type           =AVMEDIA_TYPE_AUDIO;
    basicframe.codec_id             =AV_CODEC_ID_PCM_MULAW;
    basicframe.subsession_index     =subsession_index;
    // prepare setup frame
    setupframe.media_type           =AVMEDIA_TYPE_AUDIO;
    setupframe.codec_id             =AV_CODEC_ID_PCM_MULAW;   // what frame types are to be expected from this stream
    setupframe.subsession_index     =subsession_index;
    setupframe.mstimestamp          =getCurrentMsTimestamp();
    // send setup frame
    framefilter.run(&setupframe);
  }
  else {
    livelogger.log(LogLevel::debug) << "FrameSink: WARNING: unknown codec "<<codec_name<<std::endl;
    basicframe.media_type           =AVMEDIA_TYPE_UNKNOWN;
    basicframe.codec_id             =AV_CODEC_ID_NONE;
    basicframe.subsession_index     =subsession_index;
    setReceiveBuffer(DEFAULT_PAYLOAD_SIZE); // sets nbuf
  }
  
#ifdef SEND_PARAMETER_SETS
  sendParameterSets();
#endif
  
  livelogger.log(LogLevel::debug) << "FrameSink: constructor: internal_frame= "<< basicframe <<std::endl;
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
  basicframe.mstimestamp=(presentationTime.tv_sec*1000+presentationTime.tv_usec/1000);
  basicframe.fillPars();
  
  basicframe.payload.resize(checkBufferSize(frameSize)); // set correct frame size .. now information about the packet length goes into the filter chain
  
  scs.setFrame(); // flag that indicates that we got a frame
    
  framefilter.run(&basicframe); // starts the frame filter chain
  
  if (numTruncatedBytes>0) {// time to grow the buffer..
    livelogger.log(LogLevel::debug) << "FrameSink : growing reserved size to "<< target_size << " bytes" << std::endl;
    setReceiveBuffer(target_size);
  }
  
  basicframe.payload.resize(basicframe.payload.capacity()); // recovers maximum size .. must set maximum size before letting live555 to write into the memory area
  
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
   if (basicframe.codec_id==AV_CODEC_ID_H264) {
     target_size+=nalstamp.size();
   }
   // target_size+=8; // receive extra mem to avoid ffmpeg decoder over-read // nopes!  This can screw up the decoder completely!
   return target_size;
}


void FrameSink::setReceiveBuffer(unsigned target_size) {
  target_size=checkBufferSize(target_size); // correct buffer size to include nalstamp
  basicframe.payload.resize(target_size);
  if (basicframe.codec_id==AV_CODEC_ID_H264) {
    fReceiveBuffer=basicframe.payload.data()+nalstamp.size(); // pointer for the beginning of the payload (after the nalstamp)
    nbuf=basicframe.payload.size()-nalstamp.size();           // size left for actual payload (without the prepending 0001)
    std::copy(nalstamp.begin(),nalstamp.end(),basicframe.payload.begin());
  }
  else {
    fReceiveBuffer=basicframe.payload.data();
    nbuf=basicframe.payload.size();
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
  delete[] pars;
}


