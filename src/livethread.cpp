/*
 * livethread.cpp : A live555 thread
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
 *  @file    livethread.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.15.0 
 *  
 *  @brief A live555 thread
 *
 */ 

#include "livethread.h"
#include "logging.h"

// #define RECONNECT_VERBOSE   // by default, disable

using namespace std::chrono_literals;
using std::this_thread::sleep_for; 


void setLiveOutPacketBuffermaxSize(unsigned i) {
  OutPacketBuffer::maxSize = i;  
}                                               


LiveFifo::LiveFifo(const char* name, FrameFifoContext ctx) : FrameFifo(name, ctx) {
}

LiveFifo::~LiveFifo() {
}


void LiveFifo::setLiveThread(void *live_thread) { // we need the LiveThread so we can call one of its methods..
    this->live_thread=live_thread;
}


bool LiveFifo::writeCopy(Frame* f, bool wait) {
    bool do_notify=false;
    bool ok=false;
    
    if (isEmpty()) { // triggerGotFrames => gotFramesEvent => readFrameFifoTask (this one re-registers itself if there are frames in the queue - if queue empty, must be re-registered here)
        do_notify=true; 
    }
    
    ok=FrameFifo::writeCopy(f,wait);
    
    ///*
    if (ok and do_notify) {
        ((LiveThread*)live_thread)->triggerGotFrames();
    }
    //*/
    /*
     *  if (ok) {
     *    ((LiveThread*)live_thread)->triggerGotFrames();
}
*/
    
    return ok;
    
    /*
     *  if (FrameFifo::writeCopy(f,wait)) {
     *    ((LiveThread*)live_thread)->triggerNextFrame();
}
*/
}




#define TIMESTAMP_CORRECTOR // keep this always defined

/*
 * #ifdef TIMESTAMP_CORRECTOR
 * // filterchain: {FrameFilter: inputfilter} --> {TimestampFrameFilter: timestampfilter} --> {FrameFilter: framefilter}
 * Connection::Connection(UsageEnvironment& env, const std::string address, SlotNumber slot, FrameFilter& framefilter, long unsigned int msreconnect) : env(env), address(address), slot(slot), framefilter(framefilter), msreconnect(msreconnect), is_playing(false), frametimer(0), timestampfilter("timestamp_filter",&framefilter), inputfilter("input_filter",slot,&timestampfilter)
 * #else
 * // filterchain: {FrameFilter: inputfilter} --> {FrameFilter: framefilter}
 * Connection::Connection(UsageEnvironment& env, const std::string address, SlotNumber slot, FrameFilter& framefilter, long unsigned int msreconnect) : env(env), address(address), slot(slot), framefilter(framefilter), msreconnect(msreconnect), is_playing(false), frametimer(0), timestampfilter("timestamp_filter",&framefilter), inputfilter("input_filter",slot,&framefilter)
 * #endif
 */

/*
 * #ifdef TIMESTAMP_CORRECTOR
 * // filterchain: {FrameFilter: inputfilter} --> {TimestampFrameFilter: timestampfilter} --> {FrameFilter: framefilter}
 * Connection::Connection(UsageEnvironment& env, LiveConnectionContext& ctx) : env(env), ctx(ctx), is_playing(false), frametimer(0), timestampfilter("timestamp_filter",ctx.framefilter), inputfilter("input_filter",ctx.slot,&timestampfilter)
 * #else
 * // filterchain: {FrameFilter: inputfilter} --> {FrameFilter: framefilter}
 * Connection::Connection(UsageEnvironment& env, LiveConnectionContext& ctx) : env(env), ctx(ctx), is_playing(false), frametimer(0), timestampfilter("timestamp_filter",NULL), inputfilter("input_filter",slot,ctx.framefilter)
 * #endif
 * {
 *  if (ctx.msreconnect>0 and ctx.msreconnect<=Timeout::livethread) {
 *    livethreadlogger.log(LogLevel::fatal) << "Connection: constructor: your requested reconnection time is less than equal to the LiveThread timeout.  You will get problems" << std::endl;
 *  }  
 * };
 */


Connection::Connection(UsageEnvironment& env, LiveConnectionContext& ctx) : env(env), ctx(ctx), is_playing(false), frametimer(0) {
    if       (ctx.time_correction==TimeCorrectionType::none) {
        // no timestamp correction: LiveThread --> {SlotFrameFilter: inputfilter} --> ctx.framefilter
        timestampfilter    = new TimestampFrameFilter2("timestampfilter", NULL); // dummy
        repeat_sps_filter  = new RepeatH264ParsFrameFilter("repeat_sps_filter", ctx.framefilter);
        inputfilter        = new SlotFrameFilter("input_filter", ctx.slot, repeat_sps_filter);
    }
    else if  (ctx.time_correction==TimeCorrectionType::dummy) {
        // smart timestamp correction:  LiveThread --> {SlotFrameFilter: inputfilter} --> {TimestampFrameFilter2: timestampfilter} --> ctx.framefilter
        timestampfilter    = new DummyTimestampFrameFilter("dummy_timestamp_filter", ctx.framefilter);
        repeat_sps_filter  = new RepeatH264ParsFrameFilter("repeat_sps_filter", timestampfilter);
        inputfilter        = new SlotFrameFilter("input_filter", ctx.slot, repeat_sps_filter);
    }
    else { // smart corrector
        // brute-force timestamp correction: LiveThread --> {SlotFrameFilter: inputfilter} --> {DummyTimestampFrameFilter: timestampfilter} --> ctx.framefilter
        timestampfilter    = new TimestampFrameFilter2("smart_timestamp_filter", ctx.framefilter);
        repeat_sps_filter  = new RepeatH264ParsFrameFilter("repeat_sps_filter", timestampfilter);
        inputfilter        = new SlotFrameFilter("input_filter", ctx.slot, repeat_sps_filter);
    }
}


Connection::~Connection() {
    delete timestampfilter;
    delete inputfilter;
    delete repeat_sps_filter;
};

void Connection::reStartStream() {
    stopStream();
    playStream();
}

void Connection::reStartStreamIf() {
}

SlotNumber Connection::getSlot() {
    return ctx.slot;
};

bool Connection::isClosed() {
    return true;
}

void Connection::forceClose() {
}




// Outbound::Outbound(UsageEnvironment &env, FrameFifo &fifo, SlotNumber slot, const std::string adr, const unsigned short int portnum, const unsigned char ttl) : env(env), fifo(fifo), slot(slot), adr(adr), portnum(portnum), ttl(ttl) {}
Outbound::Outbound(UsageEnvironment &env, FrameFifo &fifo, LiveOutboundContext &ctx) : env(env), fifo(fifo), ctx(ctx), setup_ok(false), at_setup(false) {}
Outbound::~Outbound() {}

void Outbound::reinit() {
    setup_ok =false;
    at_setup =false;
    // deallocate session and subsessions
}

void Outbound::handleFrame(Frame* f) {
    /* 
     * 
     * The session and subsession setup/reinit logic.
     * The idea is, that we first receive N setup frames.  One for each substream.
     * Once there are no more setup frames coming, we close up the setup and start accepting payload
     * 
     * at_setup  = doing setup
     * setup_ok  = did setup and got first payload frame
     * start with at_setup, setup_ok = false
     * 
     * if (setup frame):
     *  if (setup_ok): REINIT
     *    call reinit:
     *      at_setup=false
     *      setup_ok=false
     *      deallocate
     *    
     *  if (not at_setup): INIT
     *    starting setup again..  create session
     *  
     *  create next subsession (according to subsession_index)
     *  check if subsession_index has been used .. reinit if necessary
     *  at_setup=true
     *  
     * else:
     *  if (at_setup): CLOSE SETUP
     *    were doing setup, but a payload frame arrived.  Close setup
     *    prepare everything for payload frames
     *    setup_ok=true
     *    
     *  if (not setup_ok):
     *    do nothing
     *  else:
     *    write payload
     */
    
    int subsession_index =f->subsession_index; // alias
    
    if ( subsession_index>=2) { // subsession_index too big
        return;
    }
    
    if (f->getFrameClass()==FrameClass::setup) { // SETUP FRAME
        if (setup_ok) { // REINIT
            reinit();
        } // REINIT
        
        if (at_setup==false) { // INIT
            // create Session
            at_setup=true;
        } // INIT
        
        // ** create here a Subsession into subsession_index
        // ** check first that it's not already occupied..
    }
    else { // PAYLOAD FRAME
        if (at_setup) { // CLOSE SETUP
            // ** do whatever necessary to close up the setup
            setup_ok=true; 
        } // CLOSE SETUP
        
        if (setup_ok==false) {
            // ** setup has not been started yet .. write an error message?
        }
        else {
            // ** write payload
        }
    } // PAYLOAD FRAME
}



// RTSPConnection::RTSPConnection(UsageEnvironment& env, const std::string address, SlotNumber slot, FrameFilter& framefilter, long unsigned int msreconnect) : Connection(env, address, slot, framefilter, msreconnect), livestatus(LiveStatus::none) {};

RTSPConnection::RTSPConnection(UsageEnvironment& env, LiveConnectionContext& ctx) : Connection(env, ctx), livestatus(LiveStatus::none) {};


RTSPConnection::~RTSPConnection() {
    // delete client;
}

/* default copy constructor good enough ..
 * RTSPConnection(const RTSPConnection &cp) : env(cp.env), address(cp.address), slot(cp.slot), framefilter(cp.framefilter)  { 
 * }
 */


void RTSPConnection::playStream() {
    if (is_playing) {
        livethreadlogger.log(LogLevel::debug) << "RTSPConnection : playStream : stream already playing" << std::endl;
    }
    else {
        // Here we are a part of the live555 event loop (this is called from periodicTask => handleSignals => stopStream => this method)
        livestatus=LiveStatus::pending;
        frametimer=0;
        livethreadlogger.log(LogLevel::crazy) << "RTSPConnection : playStream" << std::endl;
        client = ValkkaRTSPClient::createNew(env, ctx.address, *inputfilter, &livestatus);
        if (ctx.request_multicast)   { client->requestMulticast(); }
        if (ctx.request_tcp)         { client->requestTCP(); }
        if (ctx.recv_buffer_size>0)  { client->setRecvBufferSize(ctx.recv_buffer_size); }
        if (ctx.reordering_time>0)   { client->setReorderingTime(ctx.reordering_time); } // WARNING: in microseconds!
        livethreadlogger.log(LogLevel::debug) << "RTSPConnection : playStream : name " << client->name() << std::endl;
        client->sendDescribeCommand(ValkkaRTSPClient::continueAfterDESCRIBE);
    }
    is_playing=true; // in the sense that we have requested a play .. and that the event handlers will try to restart the play infinitely..
}


void RTSPConnection::stopStream() {
    // Medium* medium;
    // HashTable* htable;
    // Here we are a part of the live555 event loop (this is called from periodicTask => handleSignals => stopStream => this method)
    livethreadlogger.log(LogLevel::crazy) << "RTSPConnection : stopStream" << std::endl;
    if (is_playing) {
        // before the RTSPClient instance destroyed itself (!) it modified the value of livestatus
        if (livestatus==LiveStatus::closed) { // so, we need this to avoid calling Media::close on our RTSPClient instance
            livethreadlogger.log(LogLevel::debug) << "RTSPConnection : stopStream: already shut down" << std::endl;
        }
        else if (livestatus==LiveStatus::pending) { // the event-loop-callback system has not yet decided what to do with this stream ..
            livethreadlogger.log(LogLevel::debug) << "RTSPConnection : stopStream: pending" << std::endl;
            // we could do .. env.taskScheduler().unscheduleDelayedTask(...);
            // .. this callback chain exits by itself.  However, we'll get problems if we delete everything before that
            // .. this happens typically, when the DESCRIBE command has been set and we're waiting for the reply.
            // an easy solution: set the timeout (i.e. the interval we can send messages to the thread) larger than the time it takes wait for the describe response
            // but what if the user sends lots of stop commands to the signal queue ..?
            // TODO: add counter for pending events .. wait for pending events, etc .. ?
            // better idea: allow only one play/stop command per stream per handleSignals interval
            // possible to wait until handleSignals has been called
        }
        else {
            ValkkaRTSPClient::shutdownStream(client, 1); // sets LiveStatus to closed
            livethreadlogger.log(LogLevel::debug) << "RTSPConnection : stopStream: shut down" << std::endl;
        }
    }
    else {
        livethreadlogger.log(LogLevel::debug) << "RTSPConnection : stopStream : stream was not playing" << std::endl;
    }
    is_playing=false;
}


void RTSPConnection::reStartStreamIf() {
    if (ctx.msreconnect<=0) { // don't attempt to reconnect
        return;
    }
    
    if (livestatus==LiveStatus::pending) { // stream trying to connect .. waiting for tcp socket most likely
        // frametimer=frametimer+Timeout::livethread;
        return;
    }
    
    if (livestatus==LiveStatus::alive) { // alive
        if (client->scs.gotFrame()) { // there has been frames .. all is well
            client->scs.clearFrame(); // reset the watch flag
            frametimer=0;
        }
        else {
            frametimer=frametimer+Timeout::livethread;
        }
    } // alive
    else if (livestatus==LiveStatus::closed) {
        frametimer=frametimer+Timeout::livethread;
    }
    else {
        livethreadlogger.log(LogLevel::fatal) << "RTSPConnection: restartStreamIf called without client";
        return;
    }
    
    #ifdef RECONNECT_VERBOSE
    std::cout << "RTSPConnection: frametimer=" << frametimer << std::endl;
    #endif
    
    if (frametimer>=ctx.msreconnect) {
        livethreadlogger.log(LogLevel::debug) << "RTSPConnection: restartStreamIf: restart at slot " << ctx.slot << std::endl;
        if (livestatus==LiveStatus::alive) {
            stopStream();
        }
        if (livestatus==LiveStatus::closed) {
            is_playing=false; // just to get playStream running ..
            playStream();
        } // so, the stream might be left to the pending state
    }
}

bool RTSPConnection::isClosed() { // not pending or playing
    return (livestatus==LiveStatus::closed or livestatus==LiveStatus::none); // either closed or not initialized at all
}

void RTSPConnection::forceClose() {
    ValkkaRTSPClient::shutdownStream(client, 1);
}


//SDPConnection::SDPConnection(UsageEnvironment& env, const std::string address, SlotNumber slot, FrameFilter& framefilter) : Connection(env, address, slot, framefilter, 0) {};

SDPConnection::SDPConnection(UsageEnvironment& env, LiveConnectionContext& ctx) : Connection(env, ctx) {};


SDPConnection :: ~SDPConnection() {
}

void SDPConnection :: playStream() {
    // great no-brainer example! https://stackoverflow.com/questions/32475317/how-to-open-the-local-sdp-file-by-live555
    MediaSession* session = NULL;
    // MediaSubsession* subsession = NULL;
    // bool ok;
    std::string sdp;
    std::ifstream infile;
    // unsigned cc;
    scs =NULL;
    is_playing=false;
    
    infile.open(ctx.address.c_str());
    
    /* 
     * https://cboard.cprogramming.com/cplusplus-programming/69272-reading-whole-file-w-ifstream.html
     * http://en.cppreference.com/w/cpp/io/manip
     * https://stackoverflow.com/questions/7443787/using-c-ifstream-extraction-operator-to-read-formatted-data-from-a-file
     * 
     */
    
    if (infile.is_open())
    {
        infile >> std::noskipws;
        sdp.assign( std::istream_iterator<char>(infile),std::istream_iterator<char>() );
        infile.close();
        livethreadlogger.log(LogLevel::debug) << "SDPConnection: reading sdp file: " << sdp << std::endl;
    }
    else {
        livethreadlogger.log(LogLevel::fatal) << "SDPConnection: FATAL! Unable to open file " << ctx.address << std::endl;
        return;
    }
    
    session = MediaSession::createNew(env, sdp.c_str());
    if (session == NULL)
    {
        env << "SDPConnection: Failed to create a MediaSession object from the SDP description: " << env.getResultMsg() << "\n";
        return;
    }
    
    is_playing=true;
    scs =new StreamClientState();
    scs->session=session;
    scs->iter = new MediaSubsessionIterator(*scs->session);
    scs->subsession_index=0;
    // ok=true;
    while ((scs->subsession = scs->iter->next()) != NULL) 
    {
        if (!scs->subsession->initiate(0))
        {
            env << "SDPConnection: Failed to initiate the \"" << *scs->subsession << "\" subsession: " << env.getResultMsg() << "\n";
            // ok=false;
        }
        else
        {
            // subsession->sink = DummySink::createNew(*env, *subsession, filename);
            env << "SDPConnection: Creating data sink for subsession \"" << *scs->subsession << "\" \n";
            // subsession->sink= FrameSink::createNew(env, *subsession, inputfilter, cc, ctx.address.c_str());
            scs->subsession->sink= FrameSink::createNew(env, *scs, *inputfilter, ctx.address.c_str());
            if (scs->subsession->sink == NULL)
            {
                env << "SDPConnection: Failed to create a data sink for the \"" << *scs->subsession << "\" subsession: " << env.getResultMsg() << "\n";
                // ok=false;
            }
            else
            {
                // adjust receive buffer size and reordering treshold time if requested
                if (scs->subsession->rtpSource() != NULL) {
                    if (ctx.reordering_time>0) {
                        scs->subsession->rtpSource()->setPacketReorderingThresholdTime(ctx.reordering_time);
                        livelogger.log(LogLevel::normal) << "SDPConnection:  packet reordering time now " << ctx.reordering_time << " microseconds " << std::endl;
                    }
                    if (ctx.recv_buffer_size>0) {
                        int socketNum = scs->subsession->rtpSource()->RTPgs()->socketNum();
                        unsigned curBufferSize = getReceiveBufferSize(env, socketNum);
                        unsigned newBufferSize = setReceiveBufferTo  (env, socketNum, ctx.recv_buffer_size);
                        livelogger.log(LogLevel::normal) << "SDPConnection:  receiving socket size changed from " << curBufferSize << " to " << newBufferSize << std::endl;
                    }
                }
                
                scs->subsession->sink->startPlaying(*scs->subsession->rtpSource(), NULL, NULL);
            }
        }
        scs->subsession_index++;
    }
    
    /*
     *  if (ok) {
     *    is_playing=true;
}
else {
    // Medium::close(scs.session);
}
*/
}


void SDPConnection :: stopStream() {
    // Medium* medium;
    livethreadlogger.log(LogLevel::crazy) << "SDPConnection : stopStream" << std::endl;
    if (scs!=NULL) {
        scs->close();
        delete scs;
        scs=NULL;
    }
    is_playing=false;
    
}



SDPOutbound::SDPOutbound(UsageEnvironment& env, FrameFifo &fifo, LiveOutboundContext& ctx) : Outbound(env,fifo,ctx) {
    streams.resize(2, NULL); // we'll be ready for two media streams
}

void SDPOutbound::reinit() {
    setup_ok =false;
    at_setup =false;
    // deallocate session and subsessions
    for (auto it=streams.begin(); it!=streams.end(); ++it) {
        if (*it!=NULL) {
            delete *it;
            *it=NULL;
        }
    }
}

void SDPOutbound::handleFrame(Frame *f) {
    int subsession_index =f->subsession_index; // alias
    
    if (subsession_index>=streams.size()) { // subsession_index too big
        livethreadlogger.log(LogLevel::fatal) << "SDPOutbound :"<<ctx.address<<" : handleFrame :  substream index overlow : "<<subsession_index<<"/"<<streams.size()<< std::endl;
        fifo.recycle(f); // return frame to the stack - never forget this!
        return;
    }
    
    if (f->getFrameClass()==FrameClass::setup) { // SETUP FRAME
        SetupFrame* setupframe = static_cast<SetupFrame*>(f);
        
        if (setup_ok) { // REINIT
            reinit();
        } // REINIT
        
        if (at_setup==false) { // INIT
            // create Session
            at_setup=true;
        } // INIT
        
        // ** create here a Subsession into subsession_index
        // ** check first that it's not already occupied..
        if (streams[subsession_index]!=NULL) {
            livethreadlogger.log(LogLevel::debug) << "SDPOutbound:"<<ctx.address <<" : handleFrame : stream reinit at subsession " << subsession_index << std::endl;
            delete streams[subsession_index];
            streams[subsession_index]=NULL;
        }
        
        livethreadlogger.log(LogLevel::debug) << "SDPOutbound:"<<ctx.address <<" : handleFrame : registering stream to subsession index " <<subsession_index<< std::endl;
        switch (setupframe->codec_id) { // NEW_CODEC_DEV // when adding new codecs, make changes here: add relevant stream per codec
            case AV_CODEC_ID_H264:
                streams[subsession_index]=new H264Stream(env, fifo, ctx.address, ctx.portnum, ctx.ttl);
                streams[subsession_index]->startPlaying();
                break;
            default:
                //TODO: implement VoidStream
                // streams[subsession_index]=new VoidStream(env, const char* adr, unsigned short int portnum, const unsigned char ttl=255);
                break;
        } // switch
        
        fifo.recycle(f); // return frame to the stack - never forget this!
    } // SETUP FRAME
    else { // PAYLOAD FRAME
        if (at_setup) { // CLOSE SETUP
            // ** do whatever necessary to close up the setup
            setup_ok=true;
        } // CLOSE SETUP
        
        if (setup_ok==false) {
            // ** setup has not been started yet .. write an error message?
            fifo.recycle(f); // return frame to the stack - never forget this!
        }
        else { // WRITE PAYLOAD
            if (streams[subsession_index]==NULL) { // invalid subsession index
                livethreadlogger.log(LogLevel::normal) << "SDPOutbound:"<<ctx.address <<" : handleFrame : no stream was registered for " << subsession_index << std::endl;
                fifo.recycle(f); // return frame to the stack - never forget this!
            }
            else if (f->getFrameClass()==FrameClass::none) { // void frame, do nothing
                fifo.recycle(f); // return frame to the stack - never forget this!
            }
            else { // send frame
                streams[subsession_index]->handleFrame(f); // its up to the stream instance to call recycle
            } // send frame
        } // WRITE PAYLOAD
    } // PAYLOAD FRAME
}


SDPOutbound::~SDPOutbound() {
    reinit();
}



RTSPOutbound::RTSPOutbound(UsageEnvironment& env, RTSPServer &server, FrameFifo &fifo, LiveOutboundContext& ctx) : Outbound(env,fifo,ctx), server(server), media_session(NULL) {
    media_subsessions.resize(2, NULL); 
}


void RTSPOutbound::reinit() {
    setup_ok =false;
    at_setup =false;
    // deallocate session and subsessions
    /*
     *  for (auto it=media_subsessions.begin(); it!=media_subsessions.end(); ++it) {
     *    if (*it!=NULL) {
     *      delete *it;
     *it=NULL;
}
}
*/
    // this should do the trick .. ?
    if (media_session!=NULL) {
        #ifdef STREAM_SEND_DEBUG
        std::cout << "RTSPOutbound: reinit: closing media_session" << std::endl;
        #endif
        // Medium::close(media_session); // NOT like this!
        // media_session->setDoneFlag(); // that tricky inner event loop.. // nopes .. that would be the subsession
        // TODO: create ServerMediaSubsessionIterator and call setDoneFlag for each subsession
        server.removeServerMediaSession(media_session); // this is the correct way..
        // media_session=NULL;
        #ifdef STREAM_SEND_DEBUG
        std::cout << "RTSPOutbound: reinit: media_session closed" << std::endl;
        #endif
    }
    
    for (auto it=media_subsessions.begin(); it!=media_subsessions.end(); ++it) {
        *it=NULL;
    }
}


void RTSPOutbound::handleFrame(Frame *f) {
    int subsession_index =f->subsession_index; // alias
    
    if (subsession_index>=media_subsessions.size()) { // subsession_index too big
        livethreadlogger.log(LogLevel::fatal) << "RTSPOutbound :"<<ctx.address<<" : handleFrame :  substream index overlow : "<<subsession_index<<"/"<<media_subsessions.size()<< std::endl;
        fifo.recycle(f); // return frame to the stack - never forget this!
        return;
    }
    
    if (f->getFrameClass()==FrameClass::setup) { // SETUP FRAME
        SetupFrame* setupframe = static_cast<SetupFrame*>(f);
        
        if (setup_ok) { // REINIT
            livethreadlogger.log(LogLevel::debug) << "RTSPOutbound:"<<ctx.address <<" : handleFrame : stream reinit " << std::endl;
            reinit();
        } // REINIT
        
        if (at_setup==false) { // INIT
            #ifdef STREAM_SEND_DEBUG
            std::cout << "RTSPOutbound: handleFrame: creating ServerMediaSession" << std::endl;
            #endif
            // create Session
            char const* descriptionString ="Session streamed by Valkka";
            char const* stream_name       =ctx.address.c_str();
            media_session = ServerMediaSession::createNew(env, stream_name, stream_name, descriptionString);
            at_setup=true;
        } // INIT
        
        // ** create here a Subsession into subsession_index
        // ** check first that it's not already occupied..
        if (media_subsessions[subsession_index]!=NULL) {
            livethreadlogger.log(LogLevel::debug) << "RTSPOutbound:"<<ctx.address <<" : handleFrame : can't reinit substream" << std::endl;
        }
        
        switch (setupframe->codec_id) { // NEW_CODEC_DEV // when adding new codecs, make changes here: add relevant stream per codec
            case AV_CODEC_ID_H264:
                #ifdef STREAM_SEND_DEBUG
                std::cout << "RTSPOutbound: handleFrame: creating H264ServerMediaSubsession" << std::endl;
                #endif
                media_subsessions[subsession_index]=H264ServerMediaSubsession::createNew(env, fifo, false); //last: re-use-first-source
                media_session->addSubsession(media_subsessions[subsession_index]); 
                break;
                
                /*
                 *      char const* streamName = "h264ESVideoTest";
                 *      char const* inputFileName = "test.264";
                 *      ServerMediaSession* sms
                 *        = ServerMediaSession::createNew(*env, streamName, streamName,
                 *                                        descriptionString);
                 *      sms->addSubsession(H264VideoFileServerMediaSubsession
                 *                        ::createNew(*env, inputFileName, reuseFirstSource));
                 *      rtspServer->addServerMediaSession(sms);
                 * 
                 *      announceStream(rtspServer, sms, streamName, inputFileName);
                 */
                
                default:
                    //TODO: implement VoidStream
                    break;
        } // switch
        fifo.recycle(f); // return frame to the stack - never forget this!
        
    }
    else { // PAYLOAD FRAME
        if (at_setup) { // CLOSE SETUP
            #ifdef STREAM_SEND_DEBUG
            std::cout << "RTSPOutbound: handleFrame: closing setup: subsession_index=" << subsession_index << std::endl;
            #endif
            // ** do whatever necessary to close up the setup
            server.addServerMediaSession(media_session);
            setup_ok=true; 
            at_setup=false;
            #ifdef STREAM_SEND_DEBUG
            char* url = server.rtspURL(media_session);
            std::cout << "RTSPOutbound: handleFrame: stream address: " << url << std::endl;
            delete[] url;
            #endif
        } // CLOSE SETUP
        
        if (setup_ok==false) {
            #ifdef STREAM_SEND_DEBUG
            std::cout << "RTSPOutbound: handleFrame: got payload but never setup: subsession_index=" << subsession_index << std::endl;
            #endif
            // ** setup has not been started yet .. write an error message?
            fifo.recycle(f); // return frame to the stack - never forget this!
        }
        
        if (media_subsessions[subsession_index]==NULL) {
            livethreadlogger.log(LogLevel::normal) << "RTSPOutbound:"<<ctx.address <<" : handleFrame : no stream registered for " << subsession_index << std::endl;
            fifo.recycle(f); // return frame to the stack - never forget this!
        }
        else if (f->getFrameClass()==FrameClass::none) { // void frame, do nothing
            fifo.recycle(f); // return frame to the stack - never forget this!
        }
        else {
            // ** write payload
            #ifdef STREAM_SEND_DEBUG
            std::cout << "RTSPOutbound: handleFrame: payload=" << *f << std::endl;
            #endif
            
            //TODO: we don't know what live555 event loop has been up to.. it may have closed the whole subsession..!
            media_subsessions[subsession_index]->handleFrame(f);
        }
    } // PAYLOAD FRAME
    
}


RTSPOutbound::~RTSPOutbound() {
    reinit();
}



LiveThread::LiveThread(const char* name, FrameFifoContext fifo_ctx) : Thread(name), infifo(name, fifo_ctx), infilter(name, &infifo), exit_requested(false), authDB(NULL), server(NULL) {
    scheduler = BasicTaskScheduler::createNew();
    env       = BasicUsageEnvironment::createNew(*scheduler);
    eventLoopWatchVariable = 0;
    // this->slots_.resize(n_max_slots,NULL); // Reserve 256 slots!
    this->slots_.resize    (I_MAX_SLOTS+1,NULL);
    this->out_slots_.resize(I_MAX_SLOTS+1,NULL);
    
    scheduler->scheduleDelayedTask(Timeout::livethread*1000,(TaskFunc*)(LiveThread::periodicTask),(void*)this);
    
    // testing event triggers..
    event_trigger_id_hello_world   = env->taskScheduler().createEventTrigger(this->helloWorldEvent);
    event_trigger_id_frame_arrived = env->taskScheduler().createEventTrigger(this->frameArrivedEvent);
    event_trigger_id_got_frames    = env->taskScheduler().createEventTrigger(this->gotFramesEvent);
    
    infifo.setLiveThread((void*)this);
    fc=0;
}


LiveThread::~LiveThread() {
    unsigned short int i;
    Connection* connection;
    
    livethreadlogger.log(LogLevel::crazy) << "LiveThread: destructor: " << std::endl;
    
    stopCall(); // stop if not stopped ..
    
    for (std::vector<Connection*>::iterator it = slots_.begin(); it != slots_.end(); ++it) {
        connection=*it;
        if (!connection) {
        }
        else {
            livethreadlogger.log(LogLevel::crazy) << "LiveThread: destructor: connection ptr : "<< connection << std::endl;
            livethreadlogger.log(LogLevel::crazy) << "LiveThread: destructor: removing connection at slot " << connection->getSlot() << std::endl;
            delete connection;
        }
    }
    
    if (server!=NULL) {
        Medium::close(server); // cstructed and dtructed outside the event loop
    }
    
    bool deleted =env->reclaim(); // ok.. this works.  I forgot to close the RTSPServer !
    livethreadlogger.log(LogLevel::crazy) << "LiveThread: deleted BasicUsageEnvironment?: " << deleted << std::endl;
    
    if (!deleted) {
        livethreadlogger.log(LogLevel::normal) << "LiveThread: WARNING: could not delete BasicUsageEnvironment" << std::endl;
    }
    
    /* // can't do this .. the destructor is protected
     *  if (!deleted) { // die, you bastard!
     *    delete env;
}
*/
    delete scheduler; 
}
// 
void LiveThread::preRun() {
    exit_requested=false;
}

void LiveThread::postRun() {
}

void LiveThread::sendSignal(LiveSignalContext signal_ctx) {
    std::unique_lock<std::mutex> lk(this->mutex);
    this->signal_fifo.push_back(signal_ctx);
}


void LiveThread::checkAlive() {
    Connection *connection;
    for (std::vector<Connection*>::iterator it = slots_.begin(); it != slots_.end(); ++it) {
        connection=*it;
        if (connection!=NULL) {
            if (connection->is_playing) {
                connection->reStartStreamIf();
            }
        }
    }
}


void LiveThread::handlePending() {
    Connection* connection;
    auto it=pending.begin();
    while (it!=pending.end()) {
        connection=*it;
        if (connection->is_playing) { // this has been scheduled for termination, without calling stop stream
            connection->stopStream();
        }
        if (connection->isClosed()) {
            livethreadlogger.log(LogLevel::crazy) << "LiveThread: handlePending: deleting a closed stream at slot " << connection->getSlot() << std::endl;
            it=pending.erase(it);
            delete connection;
        }
        else {
            it++;
        }
    }
}


void LiveThread::closePending() { // call only after handlePending
    Connection* connection;
    for (auto it=pending.begin(); it!=pending.end(); ++it) {
        connection=*it;
        connection->forceClose();
        delete connection;
    }
}


void LiveThread::handleSignals() {
    std::unique_lock<std::mutex> lk(this->mutex);
    unsigned short int i;
    LiveConnectionContext connection_ctx;
    LiveOutboundContext   out_ctx;
    
    
    // if (signal_fifo.empty()) {return;}
    
    // handle pending signals from the signals fifo
    for (auto it = signal_fifo.begin(); it != signal_fifo.end(); ++it) { // it == pointer to the actual object (struct LiveSignalContext)
        
        switch (it->signal) {
            case LiveSignal::exit:
                
                for(i=0;i<=I_MAX_SLOTS;i++) { // stop and deregister all streams
                    connection_ctx.slot=i;
                    deregisterStream(connection_ctx);
                }
                
                for(i=0;i<=I_MAX_SLOTS;i++) { // stop and deregister all streams
                    out_ctx.slot=i;
                    deregisterOutbound(out_ctx);
                }
                
                // this->eventLoopWatchVariable=1;
                exit_requested=true;
                break;
                // inbound streams
            case LiveSignal::register_stream:
                this->registerStream(*(it->connection_context));
                break;
            case LiveSignal::deregister_stream:
                this->deregisterStream(*(it->connection_context));
                break;
            case LiveSignal::play_stream:
                this->playStream(*(it->connection_context));
                break;
            case LiveSignal::stop_stream:
                this->stopStream(*(it->connection_context));
                break;
                // outbound streams
            case LiveSignal::register_outbound:
                this->registerOutbound(*(it->outbound_context));
                break;
            case LiveSignal::deregister_outbound:
                // std::cout << "LiveThread : handleSignals : deregister_outbound" << std::endl;
                this->deregisterOutbound(*(it->outbound_context));
                break;
            default:
                std::cout << "LiveThread : handleSignals : unknown signal " << int(it->signal) << std::endl;
                break;
        }
    }
    
    signal_fifo.clear();
}


void LiveThread::handleFrame(Frame *f) { // handle an incoming frame ..
    int i;
    int subsession_index;
    Outbound* outbound;
    Stream* stream;
    
    if (safeGetOutboundSlot(f->n_slot,outbound)>0) { // got frame
        #ifdef STREAM_SEND_DEBUG
        std::cout << "LiveThread : "<< this->name <<" : handleFrame : accept frame "<<*f << std::endl;
        #endif
        outbound->handleFrame(f); // recycling handled deeper in the code
    } 
    else {
        #ifdef STREAM_SEND_DEBUG
        std::cout << "LiveThread : "<< this->name <<" : handleFrame : discard frame "<<*f << std::endl;
        #endif
        infifo.recycle(f);
    }
}


void LiveThread::run() {
    env->taskScheduler().doEventLoop(&eventLoopWatchVariable);
    livethreadlogger.log(LogLevel::debug) << this->name << " run : live555 loop exit " << std::endl;
}


/*
 * void LiveThread::resetConnectionContext_() {
 * this->connection_ctx.connection_type=LiveThread::LiveConnectionType::none;
 * this->connection_ctx.address        =std::string();
 * this->connection_ctx.slot           =0;
 * }
 */


int LiveThread::safeGetSlot(SlotNumber slot, Connection*& con) { // -1 = out of range, 0 = free, 1 = reserved // &* = modify pointer in-place
    Connection* connection;
    livethreadlogger.log(LogLevel::crazy) << "LiveThread: safeGetSlot" << std::endl;
    
    if (slot>I_MAX_SLOTS) {
        livethreadlogger.log(LogLevel::fatal) << "LiveThread: safeGetSlot: WARNING! Slot number overfow : increase I_MAX_SLOTS in sizes.h" << std::endl;
        return -1;
    }
    
    try {
        connection=this->slots_[slot];
    }
    catch (std::out_of_range) {
        livethreadlogger.log(LogLevel::debug) << "LiveThread: safeGetSlot : slot " << slot << " is out of range! " << std::endl;
        con=NULL;
        return -1;
    }
    if (!connection) {
        livethreadlogger.log(LogLevel::crazy) << "LiveThread: safeGetSlot : nothing at slot " << slot << std::endl;
        con=NULL;
        return 0;
    }
    else {
        livethreadlogger.log(LogLevel::debug) << "LiveThread: safeGetSlot : returning " << slot << std::endl;
        con=connection;
        return 1;
    }
}


int LiveThread::safeGetOutboundSlot(SlotNumber slot, Outbound*& outbound) { // -1 = out of range, 0 = free, 1 = reserved // &* = modify pointer in-place
    Outbound* out_;
    livethreadlogger.log(LogLevel::crazy) << "LiveThread: safeGetOutboundSlot" << std::endl;
    
    if (slot>I_MAX_SLOTS) {
        livethreadlogger.log(LogLevel::fatal) << "LiveThread: safeGetOutboundSlot: WARNING! Slot number overfow : increase I_MAX_SLOTS in sizes.h" << std::endl;
        return -1;
    }
    
    try {
        out_=this->out_slots_[slot];
    }
    catch (std::out_of_range) {
        livethreadlogger.log(LogLevel::debug) << "LiveThread: safeGetOutboundSlot : slot " << slot << " is out of range! " << std::endl;
        outbound=NULL;
        return -1;
    }
    if (!out_) {
        livethreadlogger.log(LogLevel::debug) << "LiveThread: safeGetOutboundSlot : nothing at slot " << slot << std::endl;
        outbound=NULL;
        return 0;
    }
    else {
        livethreadlogger.log(LogLevel::debug) << "LiveThread: safeGetOutboundSlot : returning " << slot << std::endl;
        outbound=out_;
        return 1;
    }
}


void LiveThread::registerStream(LiveConnectionContext &connection_ctx) {
    // semantics:
    // register   : create RTSP/SDPConnection object into the slots_ vector
    // play       : create RTSPClient object in the Connection object .. start the callback chain describe => play, etc.
    // stop       : start shutting down by calling shutDownStream .. destruct the RTSPClient object
    // deregister : stop (if playing), and destruct RTSP/SDPConnection object from the slots_ vector
    Connection* connection;
    livethreadlogger.log(LogLevel::crazy) << "LiveThread: registerStream" << std::endl;
    switch (safeGetSlot(connection_ctx.slot,connection)) {
        case -1: // out of range
            break;
            
        case 0: // slot is free
            switch (connection_ctx.connection_type) {
                
                case LiveConnectionType::rtsp:
                    // this->slots_[connection_ctx.slot] = new RTSPConnection(*(this->env), connection_ctx.address, connection_ctx.slot, *(connection_ctx.framefilter), connection_ctx.msreconnect);
                    this->slots_[connection_ctx.slot] = new RTSPConnection(*(this->env), connection_ctx);
                    livethreadlogger.log(LogLevel::debug) << "LiveThread: registerStream : rtsp stream registered at slot " << connection_ctx.slot << " with ptr " << this->slots_[connection_ctx.slot] << std::endl;
                    // this->slots_[connection_ctx.slot]->playStream(); // not here ..
                    break;
                    
                case LiveConnectionType::sdp:
                    // this->slots_[connection_ctx.slot] = new SDPConnection(*(this->env), connection_ctx.address, connection_ctx.slot, *(connection_ctx.framefilter));
                    this->slots_[connection_ctx.slot] = new SDPConnection(*(this->env), connection_ctx);
                    livethreadlogger.log(LogLevel::debug) << "LiveThread: registerStream : sdp stream registered at slot "  << connection_ctx.slot << " with ptr " << this->slots_[connection_ctx.slot] << std::endl;
                    // this->slots_[connection_ctx.slot]->playStream(); // not here ..
                    break;
                    
                default:
                    livethreadlogger.log(LogLevel::normal) << "LiveThread: registerStream : no such LiveConnectionType" << std::endl;
                    break;
            } // switch connection_ctx.connection_type
            
            break;
            
                case 1: // slot is reserved
                    livethreadlogger.log(LogLevel::normal) << "LiveThread: registerStream : slot " << connection_ctx.slot << " is reserved! " << std::endl;
                    break;
    } // safeGetSlot(connection_ctx.slot,connection)
    
}


void LiveThread::deregisterStream(LiveConnectionContext &connection_ctx) {
    Connection* connection;
    livethreadlogger.log(LogLevel::crazy) << "LiveThread: deregisterStream" << std::endl;
    switch (safeGetSlot(connection_ctx.slot,connection)) {
        case -1: // out of range
            break;
        case 0: // slot is free
            livethreadlogger.log(LogLevel::crazy) << "LiveThread: deregisterStream : nothing at slot " << connection_ctx.slot << std::endl;
            break;
        case 1: // slot is reserved
            livethreadlogger.log(LogLevel::debug) << "LiveThread: deregisterStream : de-registering " << connection_ctx.slot << std::endl;
            if (connection->is_playing) {
                connection->stopStream();
            }
            if (!connection->isClosed()) { // didn't close correctly .. queue for stopping
                livethreadlogger.log(LogLevel::debug) << "LiveThread: deregisterStream : queing for stopping: " << connection_ctx.slot << std::endl;
                pending.push_back(connection);
            }
            else {
                delete connection;
            }
            this->slots_[connection_ctx.slot]=NULL; // case 1
            break;
    } // switch
}


void LiveThread::playStream(LiveConnectionContext &connection_ctx) {
    Connection* connection;
    livethreadlogger.log(LogLevel::crazy) << "LiveThread: playStream" << std::endl;  
    switch (safeGetSlot(connection_ctx.slot,connection)) {
        case -1: // out of range
            break;
        case 0: // slot is free
            livethreadlogger.log(LogLevel::normal) << "LiveThread: playStream : nothing at slot " << connection_ctx.slot << std::endl;
            break;
        case 1: // slot is reserved
            livethreadlogger.log(LogLevel::debug) << "LiveThread: playStream : playing.. " << connection_ctx.slot << std::endl;
            connection->playStream();
            break;
    }
}


void LiveThread::stopStream(LiveConnectionContext &connection_ctx) {
    Connection* connection;
    livethreadlogger.log(LogLevel::crazy) << "LiveThread: stopStream" << std::endl;
    switch (safeGetSlot(connection_ctx.slot,connection)) {
        case -1: // out of range
            break;
        case 0: // slot is free
            livethreadlogger.log(LogLevel::normal) << "LiveThread: stopStream : nothing at slot " << connection_ctx.slot << std::endl;
            break;
        case 1: // slot is reserved
            livethreadlogger.log(LogLevel::debug) << "LiveThread: stopStream : stopping.. " << connection_ctx.slot << std::endl;
            connection->stopStream();
            break;
    }
}


void LiveThread::registerOutbound(LiveOutboundContext &outbound_ctx) {
    Outbound* outbound;
    switch (safeGetOutboundSlot(outbound_ctx.slot,outbound)) {
        case -1: // out of range
            break;
        case 0: // slot is free
            switch (outbound_ctx.connection_type) {
                
                case LiveConnectionType::sdp:
                    // this->out_slots_[outbound_ctx.slot] = new SDPOutbound(*env, infifo, outbound_ctx.slot, outbound_ctx.address, outbound_ctx.portnum, outbound_ctx.ttl);
                    this->out_slots_[outbound_ctx.slot] = new SDPOutbound(*env, infifo, outbound_ctx);
                    livethreadlogger.log(LogLevel::debug) << "LiveThread: "<<name<<" registerOutbound : sdp stream registered at slot "  << outbound_ctx.slot << " with ptr " << this->out_slots_[outbound_ctx.slot] << std::endl;
                    //std::cout << "LiveThread : registerOutbound : " << this->out_slots_[2] << std::endl;
                    break;
                    
                case LiveConnectionType::rtsp:
                    if (!server) {
                        livethreadlogger.log(LogLevel::fatal) << "LiveThread: registerOutbound: no RTSP server initialized" << std::endl;
                    }
                    else {
                        this->out_slots_[outbound_ctx.slot] = new RTSPOutbound(*env, *server, infifo, outbound_ctx);
                        livethreadlogger.log(LogLevel::debug) << "LiveThread: "<<name<<" registerOutbound : rtsp stream registered at slot "  << outbound_ctx.slot << " with ptr " << this->out_slots_[outbound_ctx.slot] << std::endl;
                    }
                    break;
                    
                default:
                    livethreadlogger.log(LogLevel::normal) << "LiveThread: "<<name<<" registerOutbound : no such LiveConnectionType" << std::endl;
                    break;
            } // switch outbound_ctx.connection_type
            break;
            
                case 1: // slot is reserved
                    livethreadlogger.log(LogLevel::normal) << "LiveThread: "<<name<<" registerOutbound : slot " << outbound_ctx.slot << " is reserved! " << std::endl;
                    break;
    }
}


void LiveThread::deregisterOutbound(LiveOutboundContext &outbound_ctx) {
    Outbound* outbound;
    //std::cout << "LiveThread : deregisterOutbound" << std::endl;
    //std::cout << "LiveThread : deregisterOutbound : " << this->out_slots_[2] << std::endl;
    switch (safeGetOutboundSlot(outbound_ctx.slot,outbound)) {
        case -1: // out of range
            break;
        case 0: // slot is free
            livethreadlogger.log(LogLevel::crazy) << "LiveThread: deregisterOutbound : nothing at slot " << outbound_ctx.slot << std::endl;
            break;
        case 1: // slot is reserved
            livethreadlogger.log(LogLevel::debug) << "LiveThread: deregisterOutbound : de-registering " << outbound_ctx.slot << std::endl;
            // TODO: what else?
            delete outbound;
            this->out_slots_[outbound_ctx.slot]=NULL;
            break;
    }
}


void LiveThread::periodicTask(void* cdata) {
    LiveThread* livethread = (LiveThread*)cdata;
    livethreadlogger.log(LogLevel::crazy) << "LiveThread: periodicTask" << std::endl;
    livethread->handlePending(); // remove connections that were pending closing, but are ok now
    // std::cout << "LiveThread: periodicTask: pending streams " << livethread->pending.size() << std::endl;
    // stopCall => handleSignals => loop over deregisterStream => stopStream
    // if isClosed, then delete the connection, otherwise put into the pending list
    
    if (livethread->pending.empty() and livethread->exit_requested) {
        livethreadlogger.log(LogLevel::crazy) << "LiveThread: periodicTask: exit: nothing pending" << std::endl;
        livethread->eventLoopWatchVariable=1;
    }
    else if (livethread->exit_requested) { // tried really hard to close everything in a clean way .. but sockets etc. might still be hanging 
        livethreadlogger.log(LogLevel::crazy) << "LiveThread: periodicTask: exit: closePending" << std::endl;
        livethread->closePending(); // eh.. we really hope the eventloop just exits and does nothing else: some ValkkaRTSPClient pointers have been nulled and these might be used in the callbacks
        livethread->eventLoopWatchVariable=1; 
    }
    
    if (!livethread->exit_requested) {
        livethread->checkAlive();
        livethread->handleSignals(); // WARNING: sending commands to live555 must be done within the event loop
        livethread->scheduler->scheduleDelayedTask(Timeout::livethread*1000,(TaskFunc*)(LiveThread::periodicTask),(void*)livethread); // re-schedule itself
    }
}



// *** API ***

void LiveThread::registerStreamCall(LiveConnectionContext &connection_ctx) {
    LiveSignalContext signal_ctx = {LiveSignal::register_stream, &connection_ctx, NULL};
    sendSignal(signal_ctx);
}

void LiveThread::deregisterStreamCall(LiveConnectionContext &connection_ctx) {
    LiveSignalContext signal_ctx = {LiveSignal::deregister_stream, &connection_ctx, NULL};
    sendSignal(signal_ctx);
}


void LiveThread::playStreamCall(LiveConnectionContext &connection_ctx) {
    LiveSignalContext signal_ctx = {LiveSignal::play_stream, &connection_ctx, NULL};
    sendSignal(signal_ctx);
}

void LiveThread::stopStreamCall(LiveConnectionContext &connection_ctx) {
    LiveSignalContext signal_ctx = {LiveSignal::stop_stream, &connection_ctx, NULL};
    sendSignal(signal_ctx);
}


void LiveThread::registerOutboundCall(LiveOutboundContext &outbound_ctx) {
    LiveSignalContext signal_ctx = {LiveSignal::register_outbound, NULL, &outbound_ctx};
    sendSignal(signal_ctx);
}


void LiveThread::deregisterOutboundCall(LiveOutboundContext &outbound_ctx) {
    // std::cout << "LiveThread : deregisterOutboundCall" << std::endl;
    LiveSignalContext signal_ctx = {LiveSignal::deregister_outbound, NULL, &outbound_ctx};
    sendSignal(signal_ctx);
}


void LiveThread::requestStopCall() {
    threadlogger.log(LogLevel::crazy) << "LiveThread: requestStopCall: "<< this->name <<std::endl;
    if (!this->has_thread) { return; } // thread never started
    if (stop_requested) { return; } // can be requested only once
    stop_requested = true;
    
    LiveSignalContext signal_ctx;
    signal_ctx.signal=LiveSignal::exit;
    
    threadlogger.log(LogLevel::crazy) << "LiveThread: sending exit signal "<< this->name <<std::endl;
    this->sendSignal(signal_ctx);
}



/*
 * LiveFifo &LiveThread::getFifo() {
 *  return infifo;
 * }
 */


FifoFrameFilter &LiveThread::getFrameFilter() {
    return infilter;
}


void LiveThread::setRTSPServer(int portnum) {
    authDB =NULL;
        
    server = RTSPServer::createNew(*env, portnum, authDB); // like in the test programs, this is instantiated outside the event loop
    if (server == NULL) {
        *env << "Failed to create live RTSP server: " << env->getResultMsg() << "\n";
        exit(1);
    }
    
    /* // rtsp server
     *   
     *  registerOutboundCall(LiveOutboundContext)
     *  LiveOutboundContext .. has slot number and framefilter
     *  LiveOutboundContext registered to out_slots_
     *  
     *  Frame is read from the inbound fifo .. a frame with slot 3 is found .. out_slots[3] has an Outbound instance.
     *  .. so, that should be an RTSPOutbound instance.
     *  RTSPOutbound is first in an uninitialized state .. when it gets config frames, should create server media session
     *  and add subsessions into it
     *  RTSPOutbound should be resettable as well
     *  Like this:
     *  
     *  ServerMediaSession* sms
     *      = ServerMediaSession::createNew(*env, streamName, streamName,
     *				      descriptionString);
     *    sms->addSubsession(H264VideoFileServerMediaSubsession
     *		       ::createNew(*env, inputFileName, reuseFirstSource));
     *    rtspServer->addServerMediaSession(sms);
     *    
     *    
     *  "H264VideoFileServerMediaSubsession" has "createNewStreamSource" that returns a FramedSource
     *  .. that is instantiated at constructor time .. or instantiated after rtsp negotiation?
     *  "createRTPSink" is called at rtsp negotiation with the FramedSource as its argument..?
     *  
     *  Outbound has Stream instances.  
     *  Each Stream has RTPSink (depends on the stream type, say H264VideoRTPSink), RTCPInstance, Groupsock
     *  and
     *  FramedSource and a BufferSource instance (BufferSource has an internal fifo)
     *  
     *  payload frame arrives .. RTSPOutbound .. choose corrent stream, write to Stream's BufferSource 
     *  
     *  ==> we just need to define "createNewStreamSource" and see that "createNewRTPSink" returns H264VideoRTPSink
     *  
     *  So, for SDP, we create the RTPSink, RTCPInstance & Groupsock ourselves.  For RTSP they're given by the server.
     *  
     *  
     *  Outbound::handleFrame .. when init frame arrives, calls Stream::startPlaying 
     */
}


void LiveThread::helloWorldEvent(void* clientData) {
    // this is the event identified by event_trigger_id_hello_world
    std::cout << "Hello world from a triggered event!" << std::endl;
}


void LiveThread::frameArrivedEvent(void* clientData) {
    Frame* f;
    LiveThread *thread = (LiveThread*)clientData;
    // this is the event identified by event_trigger_id_frame
    // std::cout << "LiveThread : frameArrived : New frame has arrived!" << std::endl;
    f=thread->infifo.read(1); // this should not block..
    thread->fc+=1;
    std::cout << "LiveThread: frameArrived: frame count=" << thread->fc << " : " << *f << std::endl;
    // std::cout << "LiveThread : frameArrived : frame :" << *f << std::endl;
    thread->infifo.recycle(f);
}


void LiveThread::gotFramesEvent(void* clientData) { // registers a periodic task to the event loop
    #ifdef STREAM_SEND_DEBUG
    std::cout << "LiveThread: gotFramesEvent " << std::endl;
    #endif
    LiveThread *thread = (LiveThread*)clientData;
    thread->scheduler->scheduleDelayedTask(0,(TaskFunc*)(LiveThread::readFrameFifoTask),(void*)thread); 
}


void LiveThread::readFrameFifoTask(void* clientData) {
    Frame* f;
    LiveThread *thread = (LiveThread*)clientData;
    #ifdef STREAM_SEND_DEBUG
    std::cout << "LiveThread: readFrameFifoTask: read" << std::endl;
    thread->infifo.diagnosis();
    #endif
    if (thread->infifo.isEmpty()) { // this task has been scheduled too many times .. nothing yet to read from the fifo
        std::cout << "LiveThread: readFrameFifoTask: underflow" << std::endl;
        return; 
    }
    f=thread->infifo.read(); // this blocks
    thread->fc+=1;
    #ifdef STREAM_SEND_DEBUG
    std::cout << "LiveThread: readFrameFifoTask: frame count=" << thread->fc << " : " << *f << std::endl;
    #endif
    
    thread->handleFrame(f);
    // thread->infifo.recycle(f); // recycling is handled deeper in the code
    
    ///*
    if (thread->infifo.isEmpty()) { // no more frames for now ..
    }
    else {
        thread->scheduler->scheduleDelayedTask(0,(TaskFunc*)(LiveThread::readFrameFifoTask),(void*)thread); // re-registers itself
    }
    //*/
    
}


void LiveThread::testTrigger() {
    // http://live-devel.live555.narkive.com/MSFiseCu/problem-with-triggerevent
    scheduler->triggerEvent(event_trigger_id_hello_world,(void*)(NULL));
}


void LiveThread::triggerGotFrames() {
    #ifdef STREAM_SEND_DEBUG
    std::cout << "LiveThread: triggerGotFrames" << std::endl;
    #endif
    // scheduler->triggerEvent(event_trigger_id_frame_arrived,(void*)(this));
    scheduler->triggerEvent(event_trigger_id_got_frames,(void*)(this)); 
}


