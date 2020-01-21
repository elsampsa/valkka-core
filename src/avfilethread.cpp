/*
 * avfilethread.cpp : A thread sending frames from files
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
 *  @file    avfilethread.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.16.0 
 *  
 *  @brief  A thread sending frames from files
 */ 

#include "avfilethread.h"


TestFileStream::TestFileStream(const char* filename) {
  // https://stackoverflow.com/questions/23404403/need-to-convert-h264-stream-from-annex-b-format-to-avcc-format
  // https://stackoverflow.com/questions/11543839/h-264-conversion-with-ffmpeg-from-a-rtp-stream
  // https://ffmpeg.org/pipermail/libav-user/2014-July/007143.html
  // "you don't have to create a codec context, instead must use the codec context in AVFormatContext::streams[idx]::codec" .. wtf?
  int i;
  unsigned short n;
  
  // avpkt =NULL;
  avpkt= new AVPacket();
  // av_new_packet(avpkt,1024*1024*10); // 10 MB
  // av_init_packet(avpkt);
  
  // return;
  
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga31d601155e9035d5b0e7efedc894ee49
  input_context=NULL;
  i=avformat_open_input(&input_context, filename, av_find_input_format("matroska"), NULL);
  
  // return;
  
  // i=avformat_open_input(&input_context, filename, NULL, NULL);
  
  if (i<0) {
    std::cout << "TestFileStream : my_avformat_open_input: got nothing from file "<< filename << std::endl;
    return;
  }
  else {
    std::cout << "TestFileStream : got " << i << " " << input_context << std::endl;
  }
  
  if (avformat_find_stream_info(input_context, NULL) < 0) { // this ****er reserves and AVPacket and sometimes does not free it properly => memleak .. actually, this happens only, when using av_read_frame afterwards in pull().
    std::cout << "TestFileStream: Could not find stream information" << std::endl;
  }
  
  /*
  for(i=0;i<input_context->nb_streams;i++) { // send setup frames describing the file contents
    AVCodecID codec_id          =input_context->streams[i]->codec->codec_id;
    AVMediaType media_type      =input_context->streams[avpkt->stream_index]->codec->codec_type;
    std::cout << "TestFileStream: codec_id, media_type: " << int(codec_id) << " " << int(media_type) << std::endl;
  }
  */
  
  annexb = av_bitstream_filter_init("h264_mp4toannexb");
} 
  

TestFileStream::~TestFileStream() {
  std::cout << "TestFileStream: dtor" << std::endl;
  // avformat_free_context(input_context);
  
  avformat_close_input(&input_context); // according to examples, this should be enough..
  
  // return;
  av_bitstream_filter_close(annexb);
  
  // av_free_packet(avpkt);
  // delete avpkt;
  av_packet_unref(avpkt);
  delete avpkt;
}


void TestFileStream::pull() {
  int i;
  int out_size;
  uint8_t *out;
  
  std::cout << "TestFileStream: pull:                     " <<  std::endl;
  
  // return;
  
  /*
  AVCodecID   codec_id   =input_context->streams[avpkt->stream_index]->codec->codec_id;
  AVMediaType media_type =input_context->streams[avpkt->stream_index]->codec->codec_type;
  */
  
  i=1;
  while(i>=0) {
    i=av_read_frame(input_context, avpkt); // MEMLEAK : this likes to re-reserve the avpkt which creates a memleak
    if (i>=0) {
      std::cout << "TestFileStream: pull: payload= " << int(avpkt->data[0]) << " " << int(avpkt->data[1]) << " " << int(avpkt->data[2]) << " " << int(avpkt->data[3])  << " " << int(avpkt->data[4]) << " " << int(avpkt->data[5]) << " (total " << avpkt->size << " bytes)" << std::endl;
      
      /*
      av_bitstream_filter_filter( // horrible memory leaks when using the filter - this bitstream filter is useless crap
        annexb,
        input_context->streams[0]->codec,
        NULL,
        &out,
        &out_size,
        avpkt->data,
        avpkt->size,
        avpkt->flags & AV_PKT_FLAG_KEY
      );
      */
      ///*
      avpkt->data[0]=0;
      avpkt->data[1]=0;
      avpkt->data[2]=0;
      avpkt->data[3]=1;
      out=avpkt->data;
      //*/
      
      std::cout << "TestFileStream: pull: filtered payload= " << int(out[0]) << " " << int(out[1]) << " " << int(out[2]) << " " << int(out[3])  << " " << int(out[4]) << " " << int(out[5]) << " (total " << avpkt->size << " bytes)" << std::endl;
    }
  }
}


// FileStream::FileStream(std::string filename, SlotNumber slot, FrameFilter& framefilter) : filename(filename), slot(slot), framefilter(framefilter) {
FileStream::FileStream(FileContext &ctx) : ctx(ctx) {
  int i;
  unsigned short n;
  
  avpkt= new AVPacket();
  av_init_packet(avpkt); 
  
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga31d601155e9035d5b0e7efedc894ee49
  input_context=NULL;
  i=avformat_open_input(&input_context, ctx.filename.c_str(), av_find_input_format("matroska"), NULL);
  // matroska_read_seek(input_context,-1,0,0);
  
  /*
  static int matroska_read_seek 	( 	AVFormatContext *  	s,
		int  	stream_index,
		int64_t  	timestamp,
		int  	flags 
	) 	
  */
  
  
  if (i<0) {
    filethreadlogger.log(LogLevel::fatal) << "FileStream : my_avformat_open_input: got nothing from file "<< ctx.filename << std::endl;
    state=FileState::error;
    return;
  }
  else {
    filethreadlogger.log(LogLevel::debug) << "FileStream : got " << i << " " << input_context << std::endl;
  }
  
  if (avformat_find_stream_info(input_context, NULL) < 0) {
    filethreadlogger.log(LogLevel::normal) << "Could not find stream information" << std::endl;
  }
  
  state=FileState::stop;
  duration=(long int)input_context->duration;
  
  n=0;
  for(i=0;i<input_context->nb_streams;i++) { // send setup frames describing the file contents
    AVCodecID codec_id          =input_context->streams[i]->codec->codec_id;
    AVMediaType media_type      =input_context->streams[i]->codec->codec_type;
    
    setupframe.sub_type         =SetupFrameType::stream_init;
    setupframe.codec_id         =codec_id;
    setupframe.media_type       =media_type;
    setupframe.subsession_index =n;
    setupframe.mstimestamp      =getCurrentMsTimestamp();
    setupframe.n_slot           =ctx.slot;
    // send setup frame
    // std::cout << "FileStream: sending setup frame: " << setupframe << std::endl;
    ctx.framefilter->run(&setupframe);
    ///*
    if (codec_id==AV_CODEC_ID_H264) { // TODO: check if this is in AVCC or Annex B (how?)
      filters.push_back(av_bitstream_filter_init("h264_mp4toannexb"));
    }
    else {
      filters.push_back(NULL);
    }
    // */
    n++;
  }
  
  reftime =0;
  seek(0);
} 
  

FileStream::~FileStream() {
  avformat_close_input(&input_context);
  // avformat_free_context(input_context); // not required
  // av_free_packet(avpkt); // dprecated
  av_packet_unref(avpkt);
  delete avpkt;
  ///* // let's not use memleaking bitstream filters
  for (auto it=filters.begin(); it!=filters.end(); it++) {
    if (*it!=NULL) {
      av_bitstream_filter_close(*it);
    }
  }
  //*/
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
  
#ifdef FILE_VERBOSE  
  std::cout << "FileStream : seek : seeking to " << ms_streamtime_ << std::endl;
#endif
  
  // i=av_seek_frame(input_context, 0, ms_streamtime_, 0); //  seek in stream.time_base units // last one: flags which define seeking direction and mode ..
  
  i=avformat_seek_file(input_context, 0, std::min((long int)0,ms_streamtime_-1000), ms_streamtime_,ms_streamtime_+500, 0);
  // TODO: .. what if stream has non-keyframes in the beginning .. then the thread will idle until the first keyframe is found
  
  filethreadlogger.log(LogLevel::debug) << "FileStream : av_seek_frame returned " << i << std::endl;
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
   
#ifdef FILE_VERBOSE  
  std::cout << "FileStream : seek : avpkt->pts : " << (long int)avpkt->pts << std::endl;
#endif
  state=FileState::seek;
  target_mstimestamp_=ms_streamtime_;
  frame_mstimestamp_ =(long int)avpkt->pts;
}


void FileStream::play() {
  setRefMstime(target_mstimestamp_);
  state=FileState::play;
}


void FileStream::stop() {
  state=FileState::stop;
}


long int FileStream::update(long int mstimestamp) { // update the target time .. (only when playing)
  long int timeout;
  if (state==FileState::stop) {
    return Timeout::filethread;
  }
  if (state==FileState::play or state==FileState::seek) { // when playing, target time changes ..
    target_mstimestamp_=mstimestamp-reftime;
  }
  timeout=pullNextFrame(); // for play and seek
  
  /*
  if (state==FileState::seek and (frame_mstimestamp_>=target_mstimestamp_)) {
    state=FileState::stop;
    timeout=Timeout::filethread;
  }
  */
  
  return timeout;
}


long int FileStream::pullNextFrame() {
  long int dt;
  int i;
#ifdef FILE_VERBOSE  
  std::cout << "FileStream: pullNextFrame:                     " <<  std::endl;
  std::cout << "FileStream: pullNextFrame: reftime            : " << reftime << std::endl;
  std::cout << "FileStream: pullNextFrame: target_mstimestamp_: " << target_mstimestamp_ << std::endl;   
  std::cout << "FileStream: pullNextFrame: frame_mstimestamp_ : " << frame_mstimestamp_ << std::endl;          // frame, stream time
  std::cout << "FileStream: pullNextFrame:                     " << std::endl;
#endif
  
  // target time has been reached if ..
  // frame_mstimestamp_-target_mstimestamp_==0
  // or
  // next frame_mstimestamp_-target_mstimestamp_ > 0
  
  dt=frame_mstimestamp_-target_mstimestamp_;
  if ( dt>0 ) { // so, this has been called in vain.. must wait still
#ifdef FILE_VERBOSE  
    std::cout << "FileStream: pullNextFrame: return timeout " << dt << std::endl;
#endif
    return dt;
  }
  else if (!(avpkt->data)) {
#ifdef FILE_VERBOSE  
  std::cout << "FileStream: pullNextFrame: no avpkt!" << std::endl;
#endif
  }
  else {
    // if frame_mstimestamp_==-1 .. there is no previous frame
    AVCodecContext*     codec_ctx  =input_context->streams[avpkt->stream_index]->codec;
    const int           &index      =avpkt->stream_index;
    const AVCodecID     &codec_id   =codec_ctx->codec_id;
    const AVMediaType   &media_type =codec_ctx->codec_type;
    uint8_t*            data        =avpkt->data;
    
    out_frame.reset();
    
    ///*
    if (!filters[index]) { // no filters for this codec
      out_frame.copyFromAVPacket(avpkt);
    }
    else { // there's a filter .. 
      if ( (codec_id==AV_CODEC_ID_H264) and (avpkt->size>=4) and (data[0]==0 and data[1]==0 and data[2]==0) ) { // must be annex b .. no need to use the filter
        out_frame.copyFromAVPacket(avpkt);
      }
      else { // use the filter
        out_frame.filterFromAVPacket(avpkt,codec_ctx,filters[index]);
      }
    }
    //*/
    
    /*
    if ( (codec_id==AV_CODEC_ID_H264) ) { // check for the h264 the stream type
      if (avpkt->size>=4 and (data[0]==0 and data[1]==0 and data[2]==0) ) { // this is annex b format ok
      }
      else if (avpkt->size>=4) { // to annex b
        data[0]=0; data[1]=0; data[2]=0; data[3]=1; // seems not to be this easy..
      }
    }
    out_frame.copyFromAVPacket(avpkt);
    */
      
    out_frame.n_slot    =ctx.slot;
    out_frame.codec_id  =codec_id;
    out_frame.media_type=media_type;
    
    frame_mstimestamp_=out_frame.mstimestamp; // the timestamp in stream time of the current frame
    // out_frame is in stream time.. let's fix that:
    out_frame.mstimestamp=frame_mstimestamp_+reftime;
    
#ifdef FILE_VERBOSE  
    std::cout << "FileStream: pullNextFrame: sending frame: " << out_frame << std::endl;
#endif
    ctx.framefilter->run(&out_frame); // send frame
    
    // read the next frame and save it for the next call
    i=av_read_frame(input_context, avpkt);
    if (i<0) {
      state=FileState::stop; // TODO: send an infoframe indicating that stream has finished
      return Timeout::filethread;
    }
    if (((long int)avpkt->pts)<=frame_mstimestamp_) { // same timestamp!  recurse (typical for sequences: sps, pps, keyframe)
#ifdef FILE_VERBOSE  
      std::cout << "FileStream: pullNextFrame: recurse: " << std::endl;
#endif
      dt=pullNextFrame();
    }
    else {
      if (state==FileState::seek) {
        state=FileState::stop;
      }
      dt=frame_mstimestamp_-target_mstimestamp_;
      return std::max((long int)0,dt);
    }
  }
}


FileThread::FileThread(const char* name, FrameFifoContext fifo_ctx) : Thread(name), infifo(name, fifo_ctx), infilter(name, &infifo) {
  this->slots_.resize(I_MAX_SLOTS+1,NULL);
}


FileThread::~FileThread() {
 FileStream* file_stream;
 // release file_stream objects in slots_
 stopCall();
 
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


void FileThread::sendSignal(FileSignalContext signal_ctx) {
  std::unique_lock<std::mutex> lk(this->mutex);
  this->signal_fifo.push_back(signal_ctx);
}


void FileThread::sendSignalAndWait(FileSignalContext signal_ctx) {
  std::unique_lock<std::mutex> lk(this->mutex);
  this->signal_fifo.push_back(signal_ctx);
  while (!this->signal_fifo.empty()) {
    this->condition.wait(lk); // lock is atomically released..
  }
}


/*
void FileThread::waitSeek() {
  std::unique_lock<std::mutex> lk(this->mutex);
  std::cout << "FileThread: waitSeek: count =" << count_streams_seeking << std::endl;
  if (count_streams_seeking<1) {return;}
  std::cout << "FileThread: waiting for seek condition" << std::endl;
  this->seek_condition.wait(lk);
}
*/


void FileThread::handleSignals() {
  std::unique_lock<std::mutex> lk(this->mutex);
  FileContext file_context;
  unsigned short int i;
  
  // if (signal_fifo.empty()) {return;} // nopes ..
  
  // handle pending signals from the signals fifo
  for (std::deque<FileSignalContext>::iterator it = signal_fifo.begin(); it != signal_fifo.end(); ++it) { // it == pointer to the actual object (struct SignalContext)
    
    switch (it->signal) {
      case FileSignal::exit:
        for(i=0;i<=I_MAX_SLOTS;i++) { // stop and deregister all streams
          file_context.slot=i;
          this->closeFileStream(file_context);
        }
        this->loop=false;
        break;
      case FileSignal::open_stream:
        this->openFileStream(*(it->file_context));
        break;
      case FileSignal::close_stream:
        this->closeFileStream(*(it->file_context));
        break;
      case FileSignal::seek_stream:
        this->seekFileStream(*(it->file_context));
        break;
      case FileSignal::play_stream:
        this->playFileStream(*(it->file_context));
        break;
      case FileSignal::stop_stream:
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
  // int seek_count;
  
  mstime    =getCurrentMsTimestamp();
  old_mstime=mstime;
  
  // timeout=Timeout::filethread;
  reached=false;
  loop   =true;
  while(loop) {
    timeout=Timeout::filethread; // assume default timeout
    // multiplexing file streams..
    // ideally, we'd like to have here a live555-type event loop..
    // i.e., no need to scan all active streams, but the the next frame from all
    // streams would be next in a queue
    // .. or the queue would have the next FileStream object that's about to give us a frame
    // seek_count=0;
    for(auto it=active_slots.begin(); it!=active_slots.end(); ++it) {
      fstream=slots_[*it];
      timeout=std::min(fstream->update(mstime),timeout);
      /*
      if (fstream->state==FileState::seek) {
        seek_count++;
      }
      */
    }
#ifdef FILE_VERBOSE  
    std::cout << "FileThread: run: timeout: " << timeout << std::endl;
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
    
    mstime =getCurrentMsTimestamp();
    if ( (mstime-old_mstime)>=Timeout::openglthread ) {
#ifdef FILE_VERBOSE  
      std::cout << "FileThread: run: calling handleSignals: " << timeout << std::endl;
#endif
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
      // file_stream = new FileStream(file_ctx.filename, file_ctx.slot, *file_ctx.framefilter); // constructor opens file
      file_stream = new FileStream(file_ctx); // constructor opens file
      if (file_stream->state==FileState::error) {
        delete file_stream;
        filethreadlogger.log(LogLevel::fatal) << "FileThread: openStream: could not open file " << file_ctx.filename << std::endl;
        file_ctx.status=FileState::error; // outgoing signal
        filethreadlogger.log(LogLevel::debug) << "FileThread: status "<< int(file_ctx.status) << std::endl;
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
      // count_streams_seeking++;
      // std::cout << "FileThread: seekFileStream: count_streams_seeking =" << count_streams_seeking << std::endl;
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
      file_stream->play();
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
  long int       duration;        ///< outgoing: duration of the stream                            
  long int       mstimestamp;     ///< outgoing: current position of the stream (stream time)      
  FileState      status;          ///< outgoing: status of the file                               
  */
  
  FileSignalContext signal_ctx = {FileSignal::open_stream, &file_ctx};
  sendSignalAndWait(signal_ctx);
  // std::cout << "FileThread : openFileStreamCall : status " << int(file_ctx.status) << std::endl;
}

void FileThread::closeFileStreamCall(FileContext &file_ctx) {
  FileSignalContext signal_ctx = {FileSignal::close_stream, &file_ctx};
  sendSignal(signal_ctx);
}

void FileThread::seekFileStreamCall(FileContext &file_ctx) {
  FileSignalContext signal_ctx = {FileSignal::seek_stream, &file_ctx};
  // sendSignal(signal_ctx);
  sendSignalAndWait(signal_ctx); // go through the seeking .. count_streams_seeking is modified
}

void FileThread::playFileStreamCall(FileContext &file_ctx) {
  FileSignalContext signal_ctx = {FileSignal::play_stream, &file_ctx};
  sendSignal(signal_ctx);
}

void FileThread::stopFileStreamCall(FileContext &file_ctx) {
  FileSignalContext signal_ctx = {FileSignal::stop_stream, &file_ctx};
  sendSignal(signal_ctx);
}


void FileThread::requestStopCall() {
    threadlogger.log(LogLevel::crazy) << "FileThread: requestStopCall: "<< this->name <<std::endl;
    if (!this->has_thread) { return; } // thread never started
    if (stop_requested) { return; } // can be requested only once
    stop_requested = true;
  
    FileSignalContext signal_ctx;
    signal_ctx.signal=FileSignal::exit;
  
    threadlogger.log(LogLevel::crazy) << "FileThread: sending exit signal "<< this->name <<std::endl;
    this->sendSignal(signal_ctx);
}



FifoFrameFilter &FileThread::getFrameFilter() {
  return infilter;
}

