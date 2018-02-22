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
 *  @version 0.3.0 
 *  
 *  @brief A live555 thread
 *
 */ 

#include "livethread.h"
#include "logging.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for; 



LiveFifo::LiveFifo(const char* name, unsigned short int n_stack) : FrameFifo(name, n_stack) {
}
  
LiveFifo::~LiveFifo() {
}
  

void LiveFifo::setLiveThread(void *live_thread) { // we need the LiveThread so we can call one of its methods..
  this->live_thread=live_thread;
}
 

bool LiveFifo::writeCopy(Frame* f, bool wait) {
  bool do_notify=false;
  
  if (isEmpty()) {
    do_notify=true;
  }
  
  if (FrameFifo::writeCopy(f,wait) and do_notify) {
    ((LiveThread*)live_thread)->triggerGotFrames();
  }
  
  /*
  if (FrameFifo::writeCopy(f,wait)) {
    ((LiveThread*)live_thread)->triggerNextFrame();
  }
  */
}




#define TIMESTAMP_CORRECTOR // keep this always defined

#ifdef TIMESTAMP_CORRECTOR
Connection::Connection(UsageEnvironment& env, const std::string address, SlotNumber slot, FrameFilter& framefilter) : env(env), address(address), slot(slot), framefilter(framefilter), is_playing(false), timestampfilter("timestamp_filter",&framefilter), inputfilter("input_filter",slot,&timestampfilter) {
  // filterchain: {FrameFilter: inputfilter} --> {TimestampFrameFilter: timestampfilter} --> {FrameFilter: framefilter}
};
#else
Connection::Connection(UsageEnvironment& env, const std::string address, SlotNumber slot, FrameFilter& framefilter) : env(env), address(address), slot(slot), framefilter(framefilter), is_playing(false), timestampfilter("timestamp_filter",&framefilter), inputfilter("input_filter",slot,&framefilter) {
  // filterchain: {FrameFilter: inputfilter} --> {FrameFilter: framefilter}
};
#endif


Connection::~Connection() {
};

void Connection::reStartStream() {
 stopStream();
 playStream();
}

SlotNumber Connection::getSlot() {
  return slot;
};


Outbound::Outbound(UsageEnvironment &env, FrameFifo &fifo, SlotNumber slot, const std::string adr, const unsigned short int portnum, const unsigned char ttl) : env(env), fifo(fifo), slot(slot), adr(adr), portnum(portnum), ttl(ttl) {
  
}
  
Outbound::~Outbound() {
  
}

void Outbound::handleFrame(Frame* f) { 
  int subsession_index;
  
  subsession_index=f->subsession_index;
  // info frame    : init
  // regular frame : make a copy
  if ( subsession_index>=streams.size()) { // subsession_index too big
    livethreadlogger.log(LogLevel::fatal) << "Outbound :"<<adr<<" : handleFrame :  stream slot overlow : "<<subsession_index<<"/"<<streams.size()<< std::endl;
    fifo.recycle(f); // return frame to the stack - never forget this!
  }
  else if (f->frametype==FrameType::setup) { // INIT
    if (streams[subsession_index]!=NULL) { // slot is occupied
      livethreadlogger.log(LogLevel::debug) << "Outbound:"<<adr <<" : handleFrame : stream reinit " << std::endl;
      delete streams[subsession_index];
      streams[subsession_index]=NULL;
    }
    // register a new stream
    livethreadlogger.log(LogLevel::debug) << "Outbound:"<<adr <<" : handleFrame : registering stream to subsession index " <<subsession_index<< std::endl;
    switch ( (f->setup_pars).frametype ) { // NEW_CODEC_DEV // when adding new codecs, make changes here: add relevant stream per codec
      case FrameType::h264:
        streams[subsession_index]=new H264Stream(env, fifo, adr, portnum, ttl);
        streams[subsession_index]->startPlaying();
        // TODO: for rtsp case, some of these are negotiated with the client
        break;
      default:
        //TODO: implement VoidStream
        // streams[subsession_index]=new VoidStream(env, const char* adr, unsigned short int portnum, const unsigned char ttl=255);
        break;
    } // switch
    fifo.recycle(f); // return frame to the stack - never forget this!
  } // got frame: DECODER INIT
  else if (streams[subsession_index]==NULL) { // woops, no decoder registered yet..
    livethreadlogger.log(LogLevel::normal) << "Outbound:"<<adr <<" : handleFrame : no stream registered for " << subsession_index << std::endl;
    fifo.recycle(f); // return frame to the stack - never forget this!
  }
  else if (f->frametype==FrameType::none) { // void frame, do nothing
    fifo.recycle(f); // return frame to the stack - never forget this!
  }
  else { // send frame
    streams[subsession_index]->handleFrame(f); // its up to the stream instance to call recycle
  } // send frame
}



RTSPConnection::RTSPConnection(UsageEnvironment& env, const std::string address, SlotNumber slot, FrameFilter& framefilter) : Connection(env, address, slot, framefilter) {
  livestatus=LiveStatus::none;
  // client=ValkkaRTSPClient::createNew(this->env, this->address, this->framefilter);
};

RTSPConnection::~RTSPConnection() {
  // delete client;
}

void RTSPConnection::playStream() {
  // Here we are a part of the live555 event loop (this is called from periodicTask => handleSignals => stopStream => this method)
  livethreadlogger.log(LogLevel::crazy) << "RTSPConnection : playStream" << std::endl;
  client=ValkkaRTSPClient::createNew(this->env, this->address, this->inputfilter, &livestatus);
  livethreadlogger.log(LogLevel::debug) << "RTSPConnection : playStream : name " << client->name() << std::endl;
  client->sendDescribeCommand(ValkkaRTSPClient::continueAfterDESCRIBE);
  is_playing=true; // in the sense that we have requested a play .. and that the event handlers will try to restart the play infinitely..
}

void RTSPConnection::stopStream() {
  Medium* medium;
  // HashTable* htable;
  
  // Here we are a part of the live555 event loop (this is called from periodicTask => handleSignals => stopStream => this method)
  livethreadlogger.log(LogLevel::crazy) << "RTSPConnection : stopStream" << std::endl;
  if (is_playing) {
    
    /* this does not work either .. will crasshhhh
    htable=MediaLookupTable::getTable();
    if (htable->isEmpty()) {
      livethreadlogger.log(LogLevel::debug) << "RTSPConnection : stopStream: media lookup empty" << std::endl;
    }
    */
    
    /*
    if (Medium::lookupByName(env,client->name(), medium)) { // this crashes if the Media has been shut
      ValkkaRTSPClient::shutdownStream(client, 1); // .. in that case, this crashes as well .. viva la vida555
      livethreadlogger.log(LogLevel::debug) << "RTSPConnection : stopStream: shut down" << std::endl;
    }
    */
    
    if (livestatus==LiveStatus::closed) { // so, we need this to avoid calling Media::close on our RTSPClient instance
      livethreadlogger.log(LogLevel::debug) << "RTSPConnection : stopStream: already shut down" << std::endl;
    }
    else {
      ValkkaRTSPClient::shutdownStream(client, 1); // .. in that case, this crashes as well .. viva la vida555
      livethreadlogger.log(LogLevel::debug) << "RTSPConnection : stopStream: shut down" << std::endl;
    }
    
    is_playing=false;
  }
  else {
    livethreadlogger.log(LogLevel::debug) << "RTSPConnection : stopStream : woops! stream was not playing" << std::endl;
  }
}



SDPConnection::SDPConnection(UsageEnvironment& env, const std::string address, SlotNumber slot, FrameFilter& framefilter) : Connection(env, address, slot, framefilter), session(NULL) {
};

SDPConnection :: ~SDPConnection() {
}

void SDPConnection :: playStream() {
  // great no-brainer example! https://stackoverflow.com/questions/32475317/how-to-open-the-local-sdp-file-by-live555
  // MediaSession* session = NULL;
  MediaSubsession* subsession = NULL;
  bool ok;
  
  std::string sdp;
  std::ifstream infile;
  unsigned cc;
  
  infile.open(address.c_str());
  
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
    livethreadlogger.log(LogLevel::fatal) << "SDPConnection: FATAL! Unable to open file " << address << std::endl;
    return;
  }
  
  session = MediaSession::createNew(env, sdp.c_str());
  if (session == NULL)
  {
    env << "Failed to create a MediaSession object from the SDP description: " << env.getResultMsg() << "\n";     
    return;
  }
  
  MediaSubsessionIterator iter(*session);
  cc=0;
  ok=true;
  while ((subsession = iter.next()) != NULL) 
  {
    if (!subsession->initiate (0))
    {
      env << "Failed to initiate the \"" << *subsession << "\" subsession: " << env.getResultMsg() << "\n";
      ok=false;
    }
    else
    {
      // subsession->sink = DummySink::createNew(*env, *subsession, filename);
      env << "Creating data sink for subsession \"" << *subsession << "\" \n";
      subsession->sink= FrameSink::createNew(env, *subsession, inputfilter, cc, address.c_str());
      if (subsession->sink == NULL)
      {
        env << "Failed to create a data sink for the \"" << *subsession << "\" subsession: " << env.getResultMsg() << "\n";
        ok=false;
      }
      else
      {
        subsession->sink->startPlaying(*subsession->rtpSource(), NULL, NULL);
      }
    }
    cc++;
  }
  
  if (ok) {
    is_playing=true;
  }
  else {
    Medium::close(session);
  }

}


void SDPConnection :: stopStream() {
  Medium* medium;
  
  livethreadlogger.log(LogLevel::crazy) << "SDPConnection : stopStream" << std::endl;
  if (is_playing) {
    if (Medium::lookupByName(env,session->name(), medium)) {
      Medium::close(session);
      livethreadlogger.log(LogLevel::debug) << "SDPConnection : stopStream: shut down" << std::endl;
    }
    is_playing=false;
  }
  else {
    livethreadlogger.log(LogLevel::debug) << "SDPConnection : stopStream : woops! stream was not playing" << std::endl;
  }
  
}


SDPOutbound::SDPOutbound(UsageEnvironment& env, FrameFifo &fifo, SlotNumber slot, const std::string adr, const unsigned short int portnum, const unsigned char ttl) : Outbound(env,fifo,slot,adr,portnum,ttl) {
  streams.resize(2); // we'll be ready for two media streams
}


SDPOutbound::~SDPOutbound() {
}



// LiveThread::LiveThread(const char* name, SlotNumber n_max_slots) : Thread(name), n_max_slots(n_max_slots) {
LiveThread::LiveThread(const char* name, unsigned short int n_stack, int core_id) : Thread(name, core_id), infifo(name, n_stack) {
  scheduler = BasicTaskScheduler::createNew();
  env       = BasicUsageEnvironment::createNew(*scheduler);
  eventLoopWatchVariable = 0;
  // this->slots_.resize(n_max_slots,NULL); // Reserve 256 slots!
  this->slots_.resize(I_MAX_SLOTS+1,NULL);
  this->out_slots_.resize(I_MAX_SLOTS+1,NULL);
  
  scheduler->scheduleDelayedTask(Timeouts::livethread*1000,(TaskFunc*)(LiveThread::periodicTask),(void*)this);

  // testing event triggers..
  event_trigger_id_hello_world   = env->taskScheduler().createEventTrigger(this->helloWorldEvent);
  event_trigger_id_frame_arrived = env->taskScheduler().createEventTrigger(this->frameArrivedEvent);
  event_trigger_id_got_frames    = env->taskScheduler().createEventTrigger(this->gotFramesEvent);
  
  infifo.setLiveThread((void*)this);
  fc=0;
}


LiveThread::~LiveThread() {
 delete scheduler;
 unsigned short int i;
 Connection* connection;
 // delete env;
 // release connection objects in slots_
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
}
// 
void LiveThread::preRun() {}
void LiveThread::postRun() {}

void LiveThread::sendSignal(SignalContext signal_ctx) {
  std::unique_lock<std::mutex> lk(this->mutex);
  this->signal_fifo.push_back(signal_ctx);
}


void LiveThread::handleSignals() {
  std::unique_lock<std::mutex> lk(this->mutex);
  LiveConnectionContext connection_ctx;
  unsigned short int i;
  
  // if (signal_fifo.empty()) {return;}
  
  // handle pending signals from the signals fifo
  for (std::deque<SignalContext>::iterator it = signal_fifo.begin(); it != signal_fifo.end(); ++it) { // it == pointer to the actual object (struct SignalContext)
    
    switch (it->signal) {
      case Signals::exit:
        for(i=0;i<=I_MAX_SLOTS;i++) { // stop and deregister all streams
          connection_ctx.slot=i;
          deregisterStream(connection_ctx);
        }
        this->eventLoopWatchVariable=1;
        break;
      // inbound streams
      case Signals::register_stream:
        this->registerStream(*(it->connection_context));
        break;
      case Signals::deregister_stream:
        this->deregisterStream(*(it->connection_context));
        break;
      case Signals::play_stream:
        this->playStream(*(it->connection_context));
        break;
      // outbound streams
      case Signals::register_outbound:
        this->registerOutbound(*(it->outbound_context));
        break;
      case Signals::deregister_outbound:
        this->deRegisterOutbound(*(it->outbound_context));
        break;
      // thread control
      case Signals::stop_stream:
        this->stopStream(*(it->connection_context));
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
    std::cout << "LiveThread : "<< this->name <<" : handleFrame : accept frame "<<*f << std::endl;
    outbound->handleFrame(f); // recycling handled deeper in the code
  } 
  else {
    std::cout << "LiveThread : "<< this->name <<" : handleFrame : discard frame "<<*f << std::endl;
    infifo.recycle(f);
  }
}


void LiveThread::run() {
  env->taskScheduler().doEventLoop(&eventLoopWatchVariable);
  livethreadlogger.log(LogLevel::debug) << this->name << " run : live555 loop exit " << std::endl;
}


/*
void LiveThread::resetConnectionContext_() {
 this->connection_ctx.connection_type=LiveThread::LiveConnectionType::none;
 this->connection_ctx.address        =std::string();
 this->connection_ctx.slot           =0;
}
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
  Connection* connection;
  livethreadlogger.log(LogLevel::crazy) << "LiveThread: registerStream" << std::endl;
  switch (safeGetSlot(connection_ctx.slot,connection)) {
    case -1: // out of range
      break;
      
    case 0: // slot is free
      switch (connection_ctx.connection_type) {
        
        case LiveConnectionType::rtsp:
          this->slots_[connection_ctx.slot] = new RTSPConnection(*(this->env), connection_ctx.address, connection_ctx.slot, *(connection_ctx.framefilter));
          livethreadlogger.log(LogLevel::debug) << "LiveThread: registerStream : rtsp stream registered at slot " << connection_ctx.slot << " with ptr " << this->slots_[connection_ctx.slot] << std::endl;
          break;
          
        case LiveConnectionType::sdp:
          this->slots_[connection_ctx.slot] = new SDPConnection(*(this->env), connection_ctx.address, connection_ctx.slot, *(connection_ctx.framefilter));
          livethreadlogger.log(LogLevel::debug) << "LiveThread: registerStream : sdp stream registered at slot "  << connection_ctx.slot << " with ptr " << this->slots_[connection_ctx.slot] << std::endl;
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
      delete connection;
      this->slots_[connection_ctx.slot]=NULL;
  }
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
      // connection->is_playing=true; // nopes!  connection sets this value by itself (depending if connection could be played or not)
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
        case LiveConnectionType::rtsp:
          livethreadlogger.log(LogLevel::fatal) << "Outbound RTSP not implemented!" << std::endl;
          break;
          
        case LiveConnectionType::sdp:
          this->out_slots_[outbound_ctx.slot] = new SDPOutbound(*env, infifo, outbound_ctx.slot, outbound_ctx.address, outbound_ctx.portnum, outbound_ctx.ttl);
          livethreadlogger.log(LogLevel::debug) << "LiveThread: "<<name<<" registerOutbound : sdp stream registered at slot "  << outbound_ctx.slot << " with ptr " << this->out_slots_[outbound_ctx.slot] << std::endl;
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


void LiveThread::deRegisterOutbound(LiveOutboundContext &outbound_ctx) {
  Outbound* outbound;
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
      if (connection->is_playing) {
        connection->stopStream();
      }
      else {
        livethreadlogger.log(LogLevel::debug) << "LiveThread: stopStream : woops slot " << connection_ctx.slot << " was not playing " << std::endl;
      }
      break;
  }
}


void LiveThread::periodicTask(void* cdata) {
  LiveThread* livethread = (LiveThread*)cdata;
  livethreadlogger.log(LogLevel::crazy) << "LiveThread: periodicTask" << std::endl;
  livethread->handleSignals(); // WARNING: sending commands to live555 must be done within the event loop
  livethread->scheduler->scheduleDelayedTask(Timeouts::livethread*1000,(TaskFunc*)(LiveThread::periodicTask),(void*)livethread); // re-schedule itself
}



// *** API ***

void LiveThread::registerStreamCall(LiveConnectionContext &connection_ctx) {
  SignalContext signal_ctx = {Signals::register_stream, &connection_ctx, NULL};
  sendSignal(signal_ctx);
}

void LiveThread::deregisterStreamCall(LiveConnectionContext &connection_ctx) {
  SignalContext signal_ctx = {Signals::deregister_stream, &connection_ctx, NULL};
  sendSignal(signal_ctx);
}

void LiveThread::playStreamCall(LiveConnectionContext &connection_ctx) {
  SignalContext signal_ctx = {Signals::play_stream, &connection_ctx, NULL};
  sendSignal(signal_ctx);
}

void LiveThread::stopStreamCall(LiveConnectionContext &connection_ctx) {
  SignalContext signal_ctx = {Signals::stop_stream, &connection_ctx, NULL};
  sendSignal(signal_ctx);
}


void LiveThread::registerOutboundCall(LiveOutboundContext &outbound_ctx) {
  SignalContext signal_ctx = {Signals::register_outbound, NULL, &outbound_ctx};
  sendSignal(signal_ctx);
}


void LiveThread::deRegisterOutboundCall(LiveOutboundContext &outbound_ctx) {
  SignalContext signal_ctx = {Signals::deregister_outbound, NULL, &outbound_ctx};
  sendSignal(signal_ctx);
}


void LiveThread::stopCall() {
  SignalContext signal_ctx;
  
  signal_ctx.signal=Signals::exit;
  sendSignal(signal_ctx);
  this->closeThread();
  this->has_thread=false;
}


LiveFifo &LiveThread::getFifo() {
  return infifo;
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
  LiveThread *thread = (LiveThread*)clientData;
  thread->scheduler->scheduleDelayedTask(0,(TaskFunc*)(LiveThread::readFrameFifoTask),(void*)thread); 
}


void LiveThread::readFrameFifoTask(void* clientData) {
  Frame* f;
  LiveThread *thread = (LiveThread*)clientData;
  // this is the event identified by event_trigger_id_frame
  // std::cout << "LiveThread : frameArrived : New frame has arrived!" << std::endl;
  f=thread->infifo.read(); // this should not block..
  thread->fc+=1;
  std::cout << "LiveThread: readFrameFifoTask: frame count=" << thread->fc << " : " << *f << std::endl;
  // std::cout << "LiveThread : frameArrived : frame :" << *f << std::endl;
  
  thread->handleFrame(f);
  // thread->infifo.recycle(f); // recycling is handled deeper in the code
  
  if (thread->infifo.isEmpty()) { // no more frames for now ..
  }
  else {
    thread->scheduler->scheduleDelayedTask(0,(TaskFunc*)(LiveThread::readFrameFifoTask),(void*)thread); // re-registers itself
  }
}


void LiveThread::testTrigger() {
  // http://live-devel.live555.narkive.com/MSFiseCu/problem-with-triggerevent
  scheduler->triggerEvent(event_trigger_id_hello_world,(void*)(NULL));
}


void LiveThread::triggerGotFrames() {
  // scheduler->triggerEvent(event_trigger_id_frame_arrived,(void*)(this));
  scheduler->triggerEvent(event_trigger_id_got_frames,(void*)(this)); 
}
  

