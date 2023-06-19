/*
 * server.cpp : Live555 interface for server side: streaming to udp sockets directly or by using an on-demand rtsp server
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
 *  @file    server.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.3.6 
 *  
 *  @brief   Live555 interface for server side: streaming to udp sockets directly or by using an on-demand rtsp server
 */

#include "tools.h"
#include "liveserver.h"

BufferSource::BufferSource(UsageEnvironment &env, FrameFifo &fifo, Boolean &canary, unsigned preferredFrameSize, unsigned playTimePerFrame, unsigned offset) : FramedSource(env), fifo(fifo), canary(canary), fPreferredFrameSize(preferredFrameSize), fPlayTimePerFrame(playTimePerFrame), offset(offset), active(true), prevtime(0)
{
#ifdef STREAM_SEND_DEBUG
    std::cout << "BufferSource: constructor!" << std::endl;
#endif
    canary = true;
}

BufferSource::~BufferSource()
{
#ifdef STREAM_SEND_DEBUG
    std::cout << "BufferSource: destructor!" << std::endl;
#endif
    for (auto it = internal_fifo.begin(); it != internal_fifo.end(); ++it)
    {
        fifo.recycle(*it);
    }

    canary = false;
}

void BufferSource::handleFrame(Frame *f)
{
#ifdef STREAM_SEND_DEBUG
    std::cout << "BufferSource : handleFrame: " << *f << std::endl;
    std::cout << "BufferSource : active     : " << int(active) << std::endl;
    std::cout << "BufferSource : waiting?   : " << int(isCurrentlyAwaitingData()) << std::endl;
#endif

    if (f->getFrameClass() != FrameClass::basic)
    { // just accept BasicFrame(s)
        fifo.recycle(f);
        return;
    }

    { // MUTEX (internal_fifo)
        // std::unique_lock<std::mutex> lk(this->mutex); // we don't need this
        // std::cout << "BufferSource: IN : " << *f << std::endl;
        internal_fifo.push_front(static_cast<BasicFrame *>(f));
    } // MUTEX (internal_fifo)
    // if (!active and isCurrentlyAwaitingData()) { // time to activate and re-schedule doGetNextFrame
    if (!active)
    {
#ifdef STREAM_SEND_DEBUG
        std::cout << "BufferSource : evoking doGetNextFrame" << std::endl;
#endif
        doGetNextFrame();
    }
    /*
    else if (internal_fifo.size() > 5) { // DEBUG : force feeding
        doGetNextFrame();
    }
    */
}

void BufferSource::doGetNextFrame()
{
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
    BasicFrame *f;
    unsigned fMaxSize_;

#ifdef STREAM_SEND_DEBUG
    std::cout << "BufferSource : doGetNextFrame : " << std::endl;
#endif

    /*
    if (!isCurrentlyAwaitingData()) {
        std::cout << "\nBufferSource : NOT AWAITING \n";
    }
    */

    { // MUTEX (internal_fifo)
        // std::unique_lock<std::mutex> lk(this->mutex); // not needed

        if (internal_fifo.empty())
        {
#ifdef STREAM_SEND_DEBUG
            std::cout << "BufferSource : doGetNextFrame : fifo empty (1) " << std::endl;
#endif
            active = false; // this method is not re-scheduled anymore .. must be called again.
            return;
        }
        active = true; // this will be re-scheduled automatically

        f = internal_fifo.back(); // ref to the last element
        internal_fifo.pop_back(); // remove the last element

        if (internal_fifo.empty())
        {
#ifdef STREAM_SEND_DEBUG
            std::cout << "BufferSource : doGetNextFrame : fifo will be empty " << std::endl;
#endif
            fDurationInMicroseconds = 0;
        }
        else
        { // approximate when this will be called again
#ifdef STREAM_SEND_DEBUG
            std::cout << "BufferSource : doGetNextFrame : more frames in fifo " << std::endl;
#endif
            fDurationInMicroseconds = (unsigned)(std::max((long int)0, internal_fifo.back()->mstimestamp - f->mstimestamp)) * 1000;
#ifdef STREAM_SEND_DEBUG
            std::cout << "BufferSource : duration = " << fDurationInMicroseconds << std::endl;
#endif
        }
    } // MUTEX (internal_fifo)

    // offset = 0; // .. i.e. we wan't the NALstamp
    fFrameSize = (unsigned)f->payload.size() - offset;

    if (fMaxSize >= fFrameSize)
    { // unsigned numbers ..
        fNumTruncatedBytes = 0;
    }
    else
    {
        fNumTruncatedBytes = fFrameSize - fMaxSize;
    }

    // std::cout << "BufferSource : awaiting : " << int(isCurrentlyAwaitingData()) << std::endl;
    // std::cout << "BufferSource : doGetNextFrame : fFrameSize = " << fFrameSize << " fNumTruncatedBytes = " << fNumTruncatedBytes << " fMaxSize = " << fMaxSize << std::endl;

#ifdef STREAM_SEND_DEBUG
    std::cout << "BufferSource : doGetNextFrame : frame        " << *f << std::endl;
    std::cout << "BufferSource : doGetNextFrame : fMaxSize     " << fMaxSize << std::endl;
    std::cout << "BufferSource : doGetNextFrame : payload size " << f->payload.size() << std::endl;
    std::cout << "BufferSource : doGetNextFrame : fFrameSize   " << fFrameSize << std::endl;
    std::cout << "BufferSource : doGetNextFrame : fNumTruncB   " << fNumTruncatedBytes << std::endl;
    std::cout << "BufferSource : doGetNextFrame : payload      " << f->dumpPayload() << std::endl;
#endif

    fFrameSize = fFrameSize - fNumTruncatedBytes;

    // fNumTruncatedBytes = 1; // hack: flag overflow although there's none

    // std::cout << "BufferSource: OUT: " << *f << std::endl << std::endl; // for end-to-end debugging

    fPresentationTime = msToTimeval(f->mstimestamp); // timestamp to time struct
    // prevtime = f->mstimestamp;

    // fPresentationTime.tv_sec   =(fMstimestamp/1000); // secs
    // fPresentationTime.tv_usec  =(fMstimestamp-fPresentationTime.tv_sec*1000)*1000; // microsecs
    // std::cout << "call_afterGetting: " << fPresentationTime.tv_sec << "." << fPresentationTime.tv_usec << " " << fFrameSize << "\n";

    // memcpy(fTo, f->payload.data(), f->payload.size());
    memcpy(fTo, f->payload.data() + offset, fFrameSize);

#ifdef STREAM_SEND_DEBUG
    std::cout << "BufferSource : doGetNextFrame : recycle     " << *f << std::endl;
#endif
    fifo.recycle(f); // return the frame to the main live555 incoming fifo
#ifdef STREAM_SEND_DEBUG
    fifo.diagnosis();
#endif

#ifdef STREAM_SEND_DEBUG
    std::cout << "BufferSource : doGetNextFrame : calling afterGetting\n";
#endif
    FramedSource::afterGetting(this); // will re-schedule BufferSource::doGetNextFrame()
}

Stream::Stream(UsageEnvironment &env, FrameFifo &fifo, const std::string adr, unsigned short int portnum, const unsigned char ttl) : env(env), fifo(fifo), buffer_source(NULL), source_alive(false)
{
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
    destinationAddress.s_addr = our_inet_addr(adr.c_str());

    Port rtpPort(portnum);
    Port rtcpPort(portnum + 1);

    // Groupsock rtpGroupsock(this->env, destinationAddress, rtpPort, ttl);
    this->rtpGroupsock = new Groupsock(this->env, destinationAddress, rtpPort, ttl);
    // this->rtpGroupsock->multicastSendOnly();

    int fd = this->rtpGroupsock->socketNum();

    // std::cout << "Stream : send buffer size = " << getSendBufferSize(this->env, fd) << std::endl;
    // increaseSendBufferTo(this->env, fd, this->nsocket); // TODO
    // increaseSendBufferTo(UsageEnvironment& env, int socket, unsigned requestedSize) //GroupSockHelper.cpp

    this->rtcpGroupsock = new Groupsock(this->env, destinationAddress, rtcpPort, ttl);
    // this->rtcpGroupsock->multicastSendOnly();
    // rtpGroupsock.multicastSendOnly(); // we're a SSM source
    // Groupsock rtcpGroupsock(*env, destinationAddress, rtcpPort, ttl);
    // rtcpGroupsock.multicastSendOnly(); // we're a SSM source

    // Create a 'H264 Video RTP' sink from the RTP 'groupsock':
    // OutPacketBuffer::maxSize = 100000;

    unsigned char CNAME[101];
    gethostname((char *)CNAME, 100);
    CNAME[100] = '\0'; // just in case ..
    memcpy(this->cname, CNAME, 101);
}

void Stream::handleFrame(Frame *f)
{
    buffer_source->handleFrame(f); // buffer source recycles the frame when ready
}

void Stream::startPlaying()
{
    sink->startPlaying(*(terminal), this->afterPlaying, this);
}

void Stream::afterPlaying(void *cdata)
{
#ifdef STREAM_SEND_DEBUG
    std::cout << "Stream: afterPlaying" << std::endl;
#endif
    Stream *stream = (Stream *)cdata;
    stream->sink->stopPlaying();
    Medium::close(stream->terminal);
}

Stream::~Stream()
{
    // std::cout << "Stream dtor" << std::endl;
    // Medium::close(buffer_source); // no effect if buffer_source==NULL.  not here!  See ~H264Stream
    delete rtpGroupsock; // nopes // why not?
    delete rtcpGroupsock; // nopes // why not?
    //delete buffer_source;
}

ValkkaServerMediaSubsession::ValkkaServerMediaSubsession(UsageEnvironment &env, FrameFifo &fifo, Boolean reuseFirstSource) : OnDemandServerMediaSubsession(env, reuseFirstSource), fifo(fifo), source_alive(false), buffer_source(NULL)
{
}
/*
See OnDemandServerMediaSubsession::deleteStream, closeStreamSource, etc.

OnDemandServerMediaSubsession is given to the RTSPServer instance.  The RTSPServer acts on the OnDemandServerMediaSubsession instances and calls
their ..

  - createNewStreamSource (called by sdpLines)
  - when teardown has been received, call deleteStream => closeStreamSource => calls Medium::close on the inputsource (that was obtained through call on createNewSource)
*/

ValkkaServerMediaSubsession::~ValkkaServerMediaSubsession()
{
}

void ValkkaServerMediaSubsession::handleFrame(Frame *f)
{
    if (!buffer_source)
    {
#ifdef STREAM_SEND_DEBUG
        std::cout << "ValkkaServerMediaSubsession: no buffer_source created yet!" << std::endl;
#endif
        fifo.recycle(f);
    }
    else if (!source_alive)
    {
#ifdef STREAM_SEND_DEBUG
        std::cout << "ValkkaServerMediaSubsession: buffer source been annihilated!" << std::endl;
#endif
        fifo.recycle(f);
        buffer_source = NULL;
        setDoneFlag();
    }
    else
    {
        buffer_source->handleFrame(f); // buffer source recycles the frame when ready
    }
}

H264Stream::H264Stream(UsageEnvironment &env, FrameFifo &fifo, const std::string adr, unsigned short int portnum, const unsigned char ttl) : Stream(env, fifo, adr, portnum, ttl)
{
    // used when stream is requested, based on an SDP file

    buffer_source = new BufferSource(env, fifo, source_alive, 0, 0, 4); // nalstamp offset: 4 // use this with H264VideoStreamDiscreteFramer
    // buffer_source  =new BufferSource(env, fifo, source_alive, 0, 0, 0);

    // http://lists.live555.com/pipermail/live-devel/2013-April/016816.html
    sink = H264VideoRTPSink::createNew(env, rtpGroupsock, 96);
    // this->rtcp      = RTCPInstance::createNew(this->env, this->rtcpGroupsock, 500,  this->cname, sink, NULL, True); // saturates the event loop!

    terminal = H264VideoStreamDiscreteFramer::createNew(env, buffer_source);
    // terminal       =H264VideoStreamFramer::createNew(env, buffer_source);
}

H264Stream::~H264Stream()
{
#ifdef STREAM_SEND_DEBUG
    std::cout << "H264Stream: destructor!" << std::endl;
#endif
    // delete sink;
    // delete terminal;
    // Medium::close(sink); // nopes
    // Medium::close(buffer_source); // nopes, because ..
    // Medium::close(terminal); // .. this should close buffer_source as well
    this->afterPlaying(this);
}

H264ServerMediaSubsession *H264ServerMediaSubsession::createNew(UsageEnvironment &env, FrameFifo &fifo, Boolean reuseFirstSource)
{
    return new H264ServerMediaSubsession(env, fifo, reuseFirstSource);
}

H264ServerMediaSubsession::H264ServerMediaSubsession(UsageEnvironment &env, FrameFifo &fifo, Boolean reuseFirstSource) : ValkkaServerMediaSubsession(env, fifo, reuseFirstSource), fAuxSDPLine(NULL), fDoneFlag(0), fDummyRTPSink(NULL)
{
}

H264ServerMediaSubsession::~H264ServerMediaSubsession()
{
#ifdef STREAM_SEND_DEBUG
    std::cout << "H264ServerMediaSubsession: destructor!" << std::endl;
#endif
    if (fAuxSDPLine != NULL)
    {
        delete[] fAuxSDPLine;
    }
    // delete buffer_source; // deleted, cause OnDemandServerMediaSubsession::closeStreamSource
}

void H264ServerMediaSubsession::afterPlayingDummy(void *clientData)
{
    H264ServerMediaSubsession *subsess = (H264ServerMediaSubsession *)clientData;
    subsess->afterPlayingDummy1();
}

void H264ServerMediaSubsession::afterPlayingDummy1()
{
    // Unschedule any pending 'checking' task:
    envir().taskScheduler().unscheduleDelayedTask(nextTask());
    // Signal the event loop that we're done:
    setDoneFlag();
    Medium::close(buffer_source);
    buffer_source = NULL;
#ifdef STREAM_SEND_DEBUG
    std::cout << "H264ServerMediaSubsession: afterPlayingDummy1: deleted buffer_source" << std::endl;
#endif
}

void H264ServerMediaSubsession::checkForAuxSDPLine(void *clientData)
{
#ifdef STREAM_SEND_DEBUG
    std::cout << "H264ServerMediaSubsession: checkForAuxSDPLine" << std::endl;
#endif
    H264ServerMediaSubsession *subsess = (H264ServerMediaSubsession *)clientData;
    subsess->checkForAuxSDPLine1();
}

void H264ServerMediaSubsession::checkForAuxSDPLine1()
{
#ifdef STREAM_SEND_DEBUG
    std::cout << "H264ServerMediaSubsession: checkForAuxSDPLine1" << std::endl;
#endif

    nextTask() = NULL;

    char const *dasl;
    if (fAuxSDPLine != NULL)
    {
        // Signal the event loop that we're done:
        setDoneFlag();
    }
    else if (fDummyRTPSink != NULL && (dasl = fDummyRTPSink->auxSDPLine()) != NULL)
    {
        fAuxSDPLine = strDup(dasl);
        fDummyRTPSink = NULL;

        // Signal the event loop that we're done:
        setDoneFlag();
    }
    else if (!fDoneFlag)
    {
        // try again after a brief delay:
        int uSecsToDelay = 100000; // 100 ms
        nextTask() = envir().taskScheduler().scheduleDelayedTask(uSecsToDelay, (TaskFunc *)checkForAuxSDPLine, this);
    }
}

char const *H264ServerMediaSubsession::getAuxSDPLine(RTPSink *rtpSink, FramedSource *inputSource)
{
#ifdef STREAM_SEND_DEBUG
    std::cout << "H264ServerMediaSubsession: getAuxSDPLine" << std::endl;
#endif
    // return ""; // TODO: on some occasions this will recurse for ever.. fix.  How to call setDoneFlag from the main program level where we can only access the OnDemandMediaServerSession-derived instance

    if (fAuxSDPLine != NULL)
        return fAuxSDPLine; // it's already been set up (for a previous client)

    if (fDummyRTPSink == NULL)
    { // we're not already setting it up for another, concurrent stream
        // Note: For H264 video files, the 'config' information ("profile-level-id" and "sprop-parameter-sets") isn't known
        // until we start reading the file.  This means that "rtpSink"s "auxSDPLine()" will be NULL initially,
        // and we need to start reading data from our file until this changes.
        fDummyRTPSink = rtpSink;

        // Start reading the file:
        fDummyRTPSink->startPlaying(*inputSource, afterPlayingDummy, this);

        // Check whether the sink's 'auxSDPLine()' is ready:
        checkForAuxSDPLine(this);
    }

    envir().taskScheduler().doEventLoop(&fDoneFlag); // tricky **it! yet another event loop.. just for reading the first bytes of the stream to generate an sdp string..

    return fAuxSDPLine;
}

FramedSource *H264ServerMediaSubsession::createNewStreamSource(unsigned /*clientSessionId*/, unsigned &estBitrate)
{
#ifdef STREAM_SEND_DEBUG
    std::cout << "H264ServerMediaSubsession: createNewStreamSource" << std::endl;
#endif
    estBitrate = 2048; // kbps, estimate

    // Create the video source:
    // buffer_source  =new BufferSource(envir(), fifo, source_alive, 0, 0, 0); // OnDemandServerMediaSubsession derived classes need the NAL stamp ..
    buffer_source = new BufferSource(envir(), fifo, source_alive, 0, 0, 4); // USE THIS! with H264VideoStreamDiscreteFramer

    // Create a framer for the Video Elementary Stream:
    // return H264VideoStreamFramer::createNew(envir(), buffer_source);
    return H264VideoStreamDiscreteFramer::createNew(envir(), buffer_source); // use this!

    // what's going on here..?
    // OnDemandServerMediaSubsession::sdpLines() calls this method .. creates a dummy RTSPSink and derives from there the sdp string
}

RTPSink *H264ServerMediaSubsession ::createNewRTPSink(Groupsock *rtpGroupsock, unsigned char rtpPayloadTypeIfDynamic, FramedSource * /*inputSource*/)
{
#ifdef STREAM_SEND_DEBUG
    std::cout << "H264ServerMediaSubsession: createNewRTPSink" << std::endl;
#endif

    // int fd = rtpGroupsock->socketNum();
    // std::cout << "H264ServerMediaSubsession : send buffer size = " << getSendBufferSize(this->envir(), fd) << std::endl; // 212992
    // increaseSendBufferTo(this->envir(), fd, 500000); // TODO
    // setSendBufferTo(this->envir(), fd, 500000); // use this
    // std::cout << "H264ServerMediaSubsession : send buffer size now = " << getSendBufferSize(this->envir(), fd) << std::endl;
    // increaseSendBufferTo(UsageEnvironment& env, int socket, unsigned requestedSize) //GroupSockHelper.cpp

    return H264VideoRTPSink::createNew(envir(), rtpGroupsock, rtpPayloadTypeIfDynamic);
}
