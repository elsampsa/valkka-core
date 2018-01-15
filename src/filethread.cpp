/*
 * filethread.cpp : A thread sending frames from files
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
 *  @file    filethread.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.2
 *  
 *  @brief  A thread sending frames from files
 */ 

#include "filethread.h"

FileStream::FileStream(std::string filename, SlotNumber slot, FrameFilter& framefilter) : filename(filename), slot(slot), framefilter(framefilter) {
  int i;
  unsigned short n;
  
  avpkt= new AVPacket();
  av_init_packet(avpkt);
  
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga31d601155e9035d5b0e7efedc894ee49
  input_context=NULL;
  i=avformat_open_input(&input_context, filename.c_str(), av_find_input_format("matroska"), NULL);
  // matroska_read_seek(input_context,-1,0,0);
  
  /*
  static int matroska_read_seek 	( 	AVFormatContext *  	s,
		int  	stream_index,
		int64_t  	timestamp,
		int  	flags 
	) 	
  */
  
  
  if (i<0) {
    std::cout << "FileStream : my_avformat_open_input: got nothing" << std::endl;
    state=FileState::error;
    return;
  }
  else {
    std::cout << "FileStream : got " << i << " " << input_context << std::endl;
  }
  
  if (avformat_find_stream_info(input_context, NULL) < 0) {
    fprintf(stderr, "Could not find stream information\n");
  }
  
  state=FileState::stop;
  duration=(long int)input_context->duration;
  
  for(i=0;i<input_context->nb_streams;i++) {
    frame_types.push_back(codec_id_to_frametype(input_context->streams[i]->codec->codec_id));
  }
  // send info frames:
  n=0;
  for(auto it=frame_types.begin(); it!=frame_types.end(); ++it) {
    setupframe.frametype=FrameType::setup; // this is a setup frame
    setupframe.setup_pars.frametype=*it;   // what frame types are to be expected from this stream
    setupframe.subsession_index=n;
    setupframe.setMsTimestamp(getCurrentMsTimestamp());
    setupframe.n_slot=slot;
    // send setup frame
    framefilter.run(&setupframe);
    n++;
  }
  reftime =0;
  frame_mstimestamp_=0;
  stream_mstimestamp_=-1;
  seek(0);
}
  

FileStream::~FileStream() {
  avformat_close_input(&input_context);
  avformat_free_context(input_context);
  av_free_packet(avpkt);
}


void FileStream::setRefMstime(long int ms_streamtime_) {
 reftime=(getCurrentMsTimestamp() - ms_streamtime_); 
}


void FileStream::seek(long int ms_streamtime_) {
  // dt=pyav.av_rescale_q(timestamp,av_time_base_q_ms,self.streams[ind].time_base)
  /* from secs to stream time units
  
  av_rescale_q(a,b,c) == a*b/c
  timestamp (in av_time_units) * (sec/av_time_unit) * (stream_unit/sec) = av_time_units * (stream_units/av_time_units)
              a                          b                  1/c
  dt=pyav.av_rescale_q(timestamp,av_time_base_q_ms,self.streams[ind].time_base)
                                  
  # lets use av time units instead ..
  timestamp (in ms) * 1 * (av_time_units/ms)
        a             b       1/c
      timestamp      unity    c=av_timebase_q_ms (ms/av_time_unit)
  */
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#gaa23f7619d8d4ea0857065d9979c75ac8
  // If stream_index is (-1), a default stream is selected, and timestamp is automatically converted from AV_TIME_BASE units to the stream specific time_base.
  int i;
  
  //state=FileState::ok;
  setRefMstime(ms_streamtime_);
  
  std::cout << "FileStream : seek : seeking to " << ms_streamtime_ << std::endl;
  
  // i=av_seek_frame(input_context, 0, ms_streamtime_, 0); //  seek in stream.time_base units // last one: flags which define seeking direction and mode ..
  
  i=avformat_seek_file(input_context, 0, std::min((long int)0,ms_streamtime_-1000), ms_streamtime_,ms_streamtime_+500, 0);
  
  filethreadlogger.log(LogLevel::normal) << "FileStream : av_seek_frame returned " << i << std::endl;
  if (i<0) {
    state=FileState::stop;
    // TODO: send an info frame indicating end
    return;
  }
  
  i=av_read_frame(input_context, avpkt);
  if (i<0) {
    state=FileState::stop;
    // TODO: send an info frame indicating stream end
    return;
  }
    
  frame_mstimestamp_=(long int)avpkt->pts;
  std::cout << "FileStream : seek : avpkt->pts : " << frame_mstimestamp_ << std::endl;
  state=FileState::seek;
  stream_mstimestamp_=-1; // no target time has not been reached yet
}


void FileStream::play() {
  setRefMstime(stream_mstimestamp_);
  state=FileState::play;
}


void FileStream::stop() {
  state=FileState::stop;
}


void FileStream::pullNextFrame(long int target_mstimestamp, long int &timeout, bool &reached) {
  long int dt;
  int i;

  ///*
  std::cout << "pullNextFrame:                     " <<  std::endl;
  std::cout << "pullNextFrame: target_mstimestamp: " << target_mstimestamp << std::endl;           // wallclock time
  std::cout << "pullNextFrame: reftime           : " << reftime << std::endl;
  std::cout << "pullNextFrame: frame_mstimestamp_: " << frame_mstimestamp_ << std::endl;          // frame, stream time
  std::cout << "pullNextFrame: frame_..+ref      : " << frame_mstimestamp_+reftime << std::endl;   // frame, wallclock time
  std::cout << "pullNextFrame: dt                : " << (frame_mstimestamp_+reftime)-target_mstimestamp << std::endl;   // frame, wallclock time
  std::cout << "pullNextFrame:                     " << std::endl;
  //*/
  
  // target time has been reached if ..
  // frame_mstimestamp-target_mstimestamp==0
  // or
  // next frame_mstimestamp-target_mstimestamp > 0
  reached=false;
  
  dt=(frame_mstimestamp_+reftime)-target_mstimestamp; // frame timestamp in wallclock time : t = t_ + reftime
  if ( dt>0 ) { // so, this has been called in vain.. must wait still
    std::cout << "pullNextFrame: return timeout " << dt << std::endl;
    timeout=dt;
    return;
  }
  else {
    if (dt==0) {
      // stream timestamp can be between the frames, but here it is exactly at a frame
      stream_mstimestamp_=frame_mstimestamp_; 
      reached=true;
    }
    out_frame.reset();
    out_frame.n_slot=slot;
    out_frame.fromAVPacket(avpkt); // copy payload, timestamp, stream index
    out_frame.frametype=frame_types[avpkt->stream_index];
    // out_frame is in stream time.. let's fix that:
    out_frame.mstimestamp=out_frame.mstimestamp+reftime;
    
    out_frame.reportMsTime(); // debugging
    
    framefilter.run(&out_frame); // send frame
    
    // read the next frame and save it for the next call
    i=av_read_frame(input_context, avpkt);
    if (i<0) {
      state=FileState::stop; // TODO: send an infoframe indicating that streams finished
      timeout=-1;
      return;
    }
    frame_mstimestamp_=(long int)avpkt->pts;
    dt=(frame_mstimestamp_+reftime)-target_mstimestamp; // frame timestamp in wallclock time : t = t_ + reftime
    if (dt>0) {
      // stream timestamp is between the frames ..
      reached=true;
      stream_mstimestamp_=frame_mstimestamp_-dt;
    }
    std::cout << "pullNextFrame: dt2               : " << dt << std::endl;
    // return std::max((long int)0,dt); // inform the caller about the timeout
    timeout=std::max((long int)0,dt);
  }
}



/** Sends all frames up to current wallclock time
 * 
 * @param target_mstimestamp  target wallclock time
 * 
 *
 *                      target_mstimestamp
 *   old (smaller value)      |                      young (bigger value)
 *       
 * 
 */
void FileStream::pullFrames(long int target_mstimestamp) { // dont use this - multiplex instead!
  int i;

  ///*
  std::cout << "pullFrames:                     " <<  std::endl;
  std::cout << "pullFrames: target_mstimestamp: " << target_mstimestamp << std::endl;           // wallclock time
  std::cout << "pullFrames: reftime           : " << reftime << std::endl;
  std::cout << "pullFrames: frame_mstimestamp_ : " << frame_mstimestamp_ << std::endl;          // frame, stream time
  std::cout << "pullFrames: frame_..+ref      : " << frame_mstimestamp_+reftime << std::endl;   // frame, wallclock time
  std::cout << "pullFrames:                     " << std::endl;
  //*/
  
  
  if ( (frame_mstimestamp_+reftime) > target_mstimestamp ) { // frame timestamp in wallclock time : t = t^(stream) + reftime
    return;
  }
  
  // TODO: how to init 
  // frame_mstimestamp_ has been saved from the previous call to pullFrames
  while( (frame_mstimestamp_+reftime) <= target_mstimestamp) {
    std::cout << "pullFrames: SENDING FRAME (stream time)      : " << frame_mstimestamp_ << std::endl;
    std::cout << "pullFrames: SENDING FRAME (diff to wallclock): " << (frame_mstimestamp_+reftime)-getCurrentMsTimestamp() << std::endl;
    
    out_frame.reset();
    out_frame.n_slot=slot;
    out_frame.fromAVPacket(avpkt); // copy payload, timestamp, stream index
    out_frame.frametype=frame_types[avpkt->stream_index];
    // out_frame is in stream time.. let's fix that:
    out_frame.mstimestamp=out_frame.mstimestamp+reftime;
    
    out_frame.reportMsTime(); // debugging
    
    framefilter.run(&out_frame);
    i=av_read_frame(input_context, avpkt);
    if (i<0) {
      state=FileState::stop;
      return;
    }
    frame_mstimestamp_=(long int)avpkt->pts;
  }
  // read the next frame and save it for the next call
  i=av_read_frame(input_context, avpkt);
  frame_mstimestamp_=(long int)avpkt->pts;
}


long int FileStream::getNextTimestamp() { // returns next frame's timestamp in wallclock time
 return frame_mstimestamp_+reftime;
}


bool FileStream::playOrSeek() {
 return (state==FileState::play or state==FileState::seek); 
}


void FileStream::stopSeek() {
  if (state==FileState::seek) {
    state==FileState::stop;
  }
}



FileThread::FileThread(const char* name, int core_id) : Thread(name, core_id) {
  this->slots_.resize(I_MAX_SLOTS+1,NULL);
}


FileThread::~FileThread() {
 FileStream* file_stream;
 // release file_stream objects in slots_
 
 for (std::vector<FileStream*>::iterator it = slots_.begin(); it != slots_.end(); ++it) {
  file_stream=*it;
  if (!file_stream) {
    }
  else {
    filethreadlogger.log(LogLevel::crazy) << "FileThread: destructor: file_stream ptr : "<< file_stream << std::endl;
    filethreadlogger.log(LogLevel::crazy) << "FileThread: destructor: removing file_stream at slot " << file_stream->getSlot() << std::endl;
    delete file_stream;
    }
 }
}

void FileThread::preRun() {}
void FileThread::postRun() {}


void FileThread::sendSignal(SignalContext signal_ctx) {
  std::unique_lock<std::mutex> lk(this->mutex);
  this->signal_fifo.push_back(signal_ctx);
}


void FileThread::sendSignalAndWait(SignalContext signal_ctx) {
  std::unique_lock<std::mutex> lk(this->mutex);
  this->signal_fifo.push_back(signal_ctx);  
  this->condition.wait(lk);
}


void FileThread::handleSignals() {
  std::unique_lock<std::mutex> lk(this->mutex);
  FileContext file_context;
  unsigned short int i;
  
  if (signal_fifo.empty()) {return;}
  
  // handle pending signals from the signals fifo
  for (std::deque<SignalContext>::iterator it = signal_fifo.begin(); it != signal_fifo.end(); ++it) { // it == pointer to the actual object (struct SignalContext)
    
    switch (it->signal) {
      case Signals::exit:
        for(i=0;i<=I_MAX_SLOTS;i++) { // stop and deregister all streams
          file_context.slot=i;
          this->closeFileStream(file_context);
        }
        this->loop=false;
        break;
      case Signals::open_stream:
        this->openFileStream(*(it->file_context));
        break;
      case Signals::close_stream:
        this->closeFileStream(*(it->file_context));
        break;
      case Signals::seek_stream:
        this->seekFileStream(*(it->file_context));
        break;
      case Signals::play_stream:
        this->playFileStream(*(it->file_context));
        break;
      case Signals::stop_stream:
        this->stopFileStream(*(it->file_context));
        break;
      }
  }
    
  signal_fifo.clear();
  condition.notify_one();
}


void FileThread::run() {
  long int mstime;
  long int old_mstime;
  long int timeout;
  long int dt;
  bool reached;
  FileStream *fstream;
  SlotNumber islot;  
  Frame* f;

  mstime    =getCurrentMsTimestamp();
  old_mstime=mstime;
  
  // timeout=Timeouts::filethread;
  start_mutex.unlock();
  
  loop=true;
  while(loop) {
    timeout=Timeouts::filethread; // assume default timeout
    ///*
    // multiplexing file streams..
    // ideally, we'd like to have here a live555-type event loop..
    // i.e., no need to scan all active streams, but the the next frame from all
    // streams would be next in a queue
    // .. or the queue would have the next FileStream object that's about to give us a frame
    for(auto it=active_slots.begin(); it!=active_slots.end(); ++it) {
      fstream=slots_[*it];
      if (fstream->playOrSeek()) { // this stream is either playing or seeking 
        fstream->pullNextFrame(mstime,dt,reached); // input: target time (mstime).  Sets timeout for next frame and a boolean flag
        std::cout << "FileThread : run : dt, reached " << dt <<" "<<reached<<std::endl;
        if (dt<0) { // there's no next frame ..
          std::cout << "FileThread : run : calling stopSeek" << std::endl;
          if (fstream->state==FileState::seek) {fstream->state=FileState::stop;}
        }
        else {
          if (fstream->state==FileState::seek and reached) { // required target time reached !
            fstream->state=FileState::stop; // if seeking, then stop
            std::cout << "FileThread : run : seeking stopped " << int(fstream->state) << std::endl;
          }
          timeout=std::min(timeout,dt);
        }
        // timeout=std::min(timeout,(slots_[*it]->getNextTimestamp()-mstime));
      }
    }
    //*/
    
    std::cout << "run: timeout: " << timeout << std::endl;
    
    // timeout=std::min(timeout,Timeouts::filethread);
    // timeout=std::max(timeout,(long int)0); // just in case ..
    // std::cout << "run: timeout2:" << timeout << std::endl;
    
    // either ..
    // sleep timeout milliseconds
    // or read from a FrameFifo with timeout millisecond timeout
    
    // http://www.martinbroadhurst.com/sleep-for-milliseconds-in-c.html
    std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
    
    mstime =getCurrentMsTimestamp();
    
    if ( (mstime-old_mstime)>=Timeouts::openglthread ) {
      handleSignals();
      old_mstime=mstime;
    }
  }
}


int FileThread::safeGetSlot(SlotNumber slot, FileStream*& con) { // -1 = out of range, 0 = free, 1 = reserved // &* = modify pointer in-place
  FileStream* file_stream;
  filethreadlogger.log(LogLevel::crazy) << "FileThread: safeGetSlot" << std::endl;
  
  if (slot>I_MAX_SLOTS) {
    filethreadlogger.log(LogLevel::fatal) << "FileThread: safeGetSlot: WARNING! Slot number overfow : increase I_MAX_SLOTS in sizes.h" << std::endl;
    return -1;
  }
  
  try {
    file_stream=this->slots_[slot];
  }
  catch (std::out_of_range) {
    filethreadlogger.log(LogLevel::debug) << "FileThread: safeGetSlot : slot " << slot << " is out of range! " << std::endl;
    con=NULL;
    return -1;
  }
  if (!file_stream) {
    filethreadlogger.log(LogLevel::debug) << "FileThread: safeGetSlot : nothing at slot " << slot << std::endl;
    con=NULL;
    return 0;
  }
  else {
    filethreadlogger.log(LogLevel::debug) << "FileThread: safeGetSlot : returning " << slot << std::endl;
    con=file_stream;
    return 1;
  }
  
}


void FileThread::openFileStream(FileContext &file_ctx) {
  FileStream* file_stream;
  filethreadlogger.log(LogLevel::crazy) << "FileThread: openStream" << std::endl;
  
  switch (safeGetSlot(file_ctx.slot,file_stream)) {
    case -1: // out of range
      break;
      
    case 0: // slot is free
      file_stream = new FileStream(file_ctx.filename, file_ctx.slot, *file_ctx.framefilter); // constructor opens file
      if (file_stream->state==FileState::error) {
        delete file_stream;
        filethreadlogger.log(LogLevel::fatal) << "FileThread: openStream: could not open file " << file_ctx.filename << std::endl;
        file_ctx.status=FileState::error; // outgoing signal
        std::cout << "FileThread: status "<< int(file_ctx.status) << std::endl;
      }
      else {
        slots_[file_ctx.slot]=file_stream;
        active_slots.push_back(file_ctx.slot);
        file_ctx.status=FileState::stop; // outgoing signal
      }
      break;
      
    case 1: // slot is reserved
      filethreadlogger.log(LogLevel::normal) << "FileThread: openStream : slot " << file_ctx.slot << " is reserved! " << std::endl;
      break;
  }
}


void FileThread::closeFileStream(FileContext &file_ctx) {
  FileStream* file_stream;
  filethreadlogger.log(LogLevel::crazy) << "FileThread: closeStream" << std::endl;
  
  switch (safeGetSlot(file_ctx.slot,file_stream)) {
    case -1: // out of range
      break;
    case 0: // slot is free
      filethreadlogger.log(LogLevel::crazy) << "FileThread: closeStream : nothing at slot " << file_ctx.slot << std::endl;
      break;
    case 1: // slot is reserved
      filethreadlogger.log(LogLevel::debug) << "FileThread: closeStream : de-registering " << file_ctx.slot << std::endl;
      delete file_stream; // destructor closes file
      this->slots_[file_ctx.slot]=NULL;
      active_slots.erase(std::find(active_slots.begin(), active_slots.end(), file_ctx.slot));
  }
}


void FileThread::seekFileStream(FileContext &file_ctx) {
  FileStream* file_stream;
  filethreadlogger.log(LogLevel::crazy) << "FileThread: seekFileStream" << std::endl;  
  
  switch (safeGetSlot(file_ctx.slot,file_stream)) {
    case -1: // out of range
      break;
    case 0: // slot is free
      filethreadlogger.log(LogLevel::crazy) << "FileThread: seekFileStream : nothing at slot " << file_ctx.slot << std::endl;
      break;
    case 1: // slot is reserved
      filethreadlogger.log(LogLevel::debug) << "FileThread: seekFileStream : seeking.. " << file_ctx.slot << std::endl;
      file_stream->seek(file_ctx.seektime_);
      break;
  }
}


void FileThread::playFileStream(FileContext &file_ctx) {
  FileStream* file_stream;
  filethreadlogger.log(LogLevel::crazy) << "FileThread: playStream" << std::endl;  
  
  switch (safeGetSlot(file_ctx.slot,file_stream)) {
    case -1: // out of range
      break;
    case 0: // slot is free
      filethreadlogger.log(LogLevel::crazy) << "FileThread: playStream : nothing at slot " << file_ctx.slot << std::endl;
      break;
    case 1: // slot is reserved
      filethreadlogger.log(LogLevel::debug) << "FileThread: playStream : playing.. " << file_ctx.slot << std::endl;
      if (file_stream->stream_mstimestamp_>0) {
        file_stream->play();
      }
      else {
        std::cout << "FileThread: can't play!  Stream time not set" << std::endl;
      }
      break;
  }
}


void FileThread::stopFileStream(FileContext &file_ctx) {
  FileStream* file_stream;
  filethreadlogger.log(LogLevel::crazy) << "FileThread: stopStream" << std::endl;
  switch (safeGetSlot(file_ctx.slot,file_stream)) {
    case -1: // out of range
      break;
    case 0: // slot is free
      filethreadlogger.log(LogLevel::crazy) << "FileThread: stopStream : nothing at slot " << file_ctx.slot << std::endl;
      break;
    case 1: // slot is reserved
      filethreadlogger.log(LogLevel::debug) << "FileThread: stopStream : stopping.. " << file_ctx.slot << std::endl;
      file_stream->stop();
      break;
  }
}


// *** API ***

void FileThread::openFileStreamCall(FileContext &file_ctx) {
  // FileContext file_ctx_;
  
  /*
  std::string    filename;        ///< incoming: the filename                                      
  SlotNumber     slot;            ///< incoming: a unique stream slot that identifies this stream 
  FrameFilter*   framefilter;     ///< incoming: the frames are feeded into this FrameFilter       
  long int       seektime_;       ///< incoming: used by signal seek_stream                        
  long int*      duration;        ///< outgoing: duration of the stream                            
  long int*      mstimestamp;     ///< outgoing: current position of the stream (stream time)      
  FileState      status;          ///< outgoing: status of the file                               
  */
  
  SignalContext signal_ctx = {Signals::open_stream, &file_ctx};
  sendSignalAndWait(signal_ctx);
  std::cout << "FileThread : openFileStreamCall : status " << int(file_ctx.status) << std::endl;
  
}

void FileThread::closeFileStreamCall(FileContext &file_ctx) {
  SignalContext signal_ctx = {Signals::close_stream, &file_ctx};
  sendSignal(signal_ctx);
}

void FileThread::seekFileStreamCall(FileContext &file_ctx) {
  SignalContext signal_ctx = {Signals::seek_stream, &file_ctx};
  sendSignal(signal_ctx);
}

void FileThread::playFileStreamCall(FileContext &file_ctx) {
  SignalContext signal_ctx = {Signals::play_stream, &file_ctx};
  sendSignal(signal_ctx);
}

void FileThread::stopFileStreamCall(FileContext &file_ctx) {
  SignalContext signal_ctx = {Signals::stop_stream, &file_ctx};
  sendSignal(signal_ctx);
}


void FileThread::stopCall() {
  SignalContext signal_ctx;
  
  signal_ctx.signal=Signals::exit;
  sendSignal(signal_ctx);
  this->closeThread();
  this->has_thread=false;
}

