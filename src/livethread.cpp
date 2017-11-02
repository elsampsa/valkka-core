/*
 * livethread.cpp : A live555 multithread
 * 
 * Copyright 2017 Sampsa Riikonen and Petri Eranko.
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Valkka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Valkka.  If not, see <http://www.gnu.org/licenses/>. 
 * 
 */

/** 
 *  @file    livethread.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
 *  
 *  @brief A live555 multithread
 *
 *  @section DESCRIPTION
 *  
 *  Yes, the description
 *
 */ 

#include "livethread.h"
#include "logging.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for; 


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



/*
void periodicTask0(void* cdata) {
  livethreadlogger.log(LogLevel::normal) << "LiveThread: periodicTask" << std::endl;
}
*/


// LiveThread::LiveThread(const char* name, SlotNumber n_max_slots) : Thread(name), n_max_slots(n_max_slots) {
LiveThread::LiveThread(const char* name, int core_id) : Thread(name, core_id) {
  scheduler = BasicTaskScheduler::createNew();
  env       = BasicUsageEnvironment::createNew(*scheduler);
  eventLoopWatchVariable = 0;
  // this->slots_.resize(n_max_slots,NULL); // Reserve 256 slots!
  this->slots_.resize(I_MAX_SLOTS+1,NULL);
  
  scheduler->scheduleDelayedTask(Timeouts::livethread*1000,(TaskFunc*)(LiveThread::periodicTask),(void*)this);
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
  
  if (signal_fifo.empty()) {return;}
  
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
      case Signals::register_stream:
        this->registerStream(it->connection_context);
        break;
      case Signals::deregister_stream:
        this->deregisterStream(it->connection_context);
        break;
      case Signals::play_stream:
        this->playStream(it->connection_context);
        break;
      case Signals::stop_stream:
        this->stopStream(it->connection_context);
        break;
      }
  }
    
  signal_fifo.clear();

}


void LiveThread::run() {
  start_mutex.unlock();
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
    livethreadlogger.log(LogLevel::debug) << "LiveThread: safeGetSlot : nothing at slot " << slot << std::endl;
    con=NULL;
    return 0;
  }
  else {
    livethreadlogger.log(LogLevel::debug) << "LiveThread: safeGetSlot : returning " << slot << std::endl;
    con=connection;
    return 1;
  }
  
}


void LiveThread::registerStream(LiveConnectionContext connection_ctx) {
  Connection* connection;
  livethreadlogger.log(LogLevel::crazy) << "LiveThread: registerStream" << std::endl;
  /*
  if (connection_ctx.connection_type==LiveConnectionType::rtsp) {
    livethreadlogger.log(LogLevel::normal) << "LiveThread: registerStream: rtsp" << std::endl;
    switch (safeGetSlot(connection_ctx.slot,connection)) {
      case -1: // out of range
        break;
      case 0: // slot is free
        this->slots_[connection_ctx.slot] = new RTSPConnection(*(this->env), connection_ctx.address, connection_ctx.slot, *(connection_ctx.framefilter));
        livethreadlogger.log(LogLevel::normal) << "LiveThread: registerStream : rtsp stream registered at slot " << connection_ctx.slot << " with ptr " << this->slots_[connection_ctx.slot] << std::endl;
        break;
      case 1: // slot is reserved
        livethreadlogger.log(LogLevel::normal) << "LiveThread: registerStream : slot " << connection_ctx.slot << " is reserved! " << std::endl;
        break;
    }
  }
  */
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


void LiveThread::deregisterStream(LiveConnectionContext connection_ctx) {
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


void LiveThread::playStream(LiveConnectionContext connection_ctx) {
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


void LiveThread::stopStream(LiveConnectionContext connection_ctx) {
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

void LiveThread::registerStreamCall(LiveConnectionContext connection_ctx) {
  SignalContext signal_ctx = {Signals::register_stream, connection_ctx};
  sendSignal(signal_ctx);
}

void LiveThread::deregisterStreamCall(LiveConnectionContext connection_ctx) {
  SignalContext signal_ctx = {Signals::deregister_stream, connection_ctx};
  sendSignal(signal_ctx);
}

void LiveThread::playStreamCall(LiveConnectionContext connection_ctx) {
  SignalContext signal_ctx = {Signals::play_stream, connection_ctx};
  sendSignal(signal_ctx);
}

void LiveThread::stopStreamCall(LiveConnectionContext connection_ctx) {
  SignalContext signal_ctx = {Signals::stop_stream, connection_ctx};
  sendSignal(signal_ctx);
}


void LiveThread::stopCall() {
  SignalContext signal_ctx;
  
  signal_ctx.signal=Signals::exit;
  sendSignal(signal_ctx);
  this->closeThread();
  this->has_thread=false;
}



