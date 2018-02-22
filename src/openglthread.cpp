/*
 * openglthread.cpp : The OpenGL thread for presenting frames and related data structures
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
 *  @file    openglthread.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.0 
 *  
 *  @brief The OpenGL thread for presenting frames and related data structures
 *
 *  @section DESCRIPTION
 *  
 */ 

#include "openglthread.h"
#include "logging.h"
#include "sizes.h"
#include "tools.h"

// WARNING: these define switches should be off (commented) by default
// #define PRESENT_VERBOSE 1 // enable this for verbose output about queing and presenting the frames in OpenGLThread // @verbosity       
// #define RENDER_VERBOSE 1 // enable this for verbose rendering
// #define NO_LATE_DROP_DEBUG 1 // don't drop late frame, but present everything in OpenGLThreads fifo.  Useful when debuggin with valgrind (as all frames arrive ~ 200 ms late)

// PFNGLXSWAPINTERVALEXTPROC glXSwapIntervalEXT = (PFNGLXSWAPINTERVALEXTPROC)glXGetProcAddress((const GLubyte*)"glXSwapIntervalEXT");  // Set the glxSwapInterval to 0, ie.

std::ostream &operator<<(std::ostream &os, OpenGLSignalContext const &m) {
 return os << "<OpenGLSignalContext: slot="<<m.n_slot<<" x_window_id="<<m.x_window_id<<" z="<<m.z<<" render_context="<<m.render_ctx<<" success="<<m.success<<">";
}


OpenGLFrameFifo::OpenGLFrameFifo(unsigned short n_stack_720p, unsigned short n_stack_1080p, unsigned short n_stack_1440p, unsigned short n_stack_4K) : FrameFifo("open_gl",0), n_stack_720p(n_stack_720p), n_stack_1080p(n_stack_1080p), n_stack_1440p(n_stack_1440p), n_stack_4K(n_stack_4K), debug(false) {

  FrameFifo::initReservoir(reservoir_720p,  n_stack_720p);
  FrameFifo::initReservoir(reservoir_1080p, n_stack_1080p);
  FrameFifo::initReservoir(reservoir_1440p, n_stack_1440p);
  FrameFifo::initReservoir(reservoir_4K,    n_stack_4K);
  // FrameFifo::initReservoir(reservoir_audio, n_stack_audio);
  
  FrameFifo::initStack(reservoir_720p,  stack_720p);
  FrameFifo::initStack(reservoir_1080p, stack_1080p);
  FrameFifo::initStack(reservoir_1440p, stack_1440p);
  FrameFifo::initStack(reservoir_4K,    stack_4K);
  // FrameFifo::initStack(reservoir_audio, stack_audio);
  
  // YUVPBO's will be reserved in OpenGLThread::preRun
}


OpenGLFrameFifo::~OpenGLFrameFifo() {
  opengllogger.log(LogLevel::debug) << "OpenGLFrameFifo: destructor "<<std::endl;
}


Frame* OpenGLFrameFifo::getFrame(BitmapType bmtype) {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  return getFrame_(bmtype);
}


Frame* OpenGLFrameFifo::getFrame_(BitmapType bmtype) {
  // std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  // .. no mutex protecting here!  This is used by writeCopy that is also mutex protected => stall
  Frame* tmpframe;
  std::deque<Frame*>* tmpstack; // alias
  
  switch(bmtype) {
    case (BitmapPars::N720::type): {
      tmpstack=&stack_720p;
      break;
    }
    case (BitmapPars::N1080::type): {
      tmpstack=&stack_1080p;
      break;
    }
    case (BitmapPars::N1440::type): {
      tmpstack=&stack_1440p;
      break;
    }
    case (BitmapPars::N4K::type): {
      tmpstack=&stack_4K;
      break;
    }
    default: {
      opengllogger.log(LogLevel::fatal) << "OpenGLFrameFifo: getFrame_: FATAL! No such bitmap type "<<bmtype<<std::endl;
      return NULL;
      break;
    }
  }
  
  if (tmpstack->empty()) {
    opengllogger.log(LogLevel::fatal) << "OpenGLFrameFifo: getFrame_: OVERFLOW! No more frames in stack for bitmap type "<<bmtype<<std::endl;
    return NULL;
  }
  
  tmpframe=(*tmpstack)[0]; // take a pointer to frame from the pre-allocated stack
  tmpstack->pop_front(); // .. remove that pointer from the stack
  
  return tmpframe;
}


/*
Frame* OpenGLFrameFifo::getAudioFrame() {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  Frame* tmpframe;
  if (stack_audio.empty()) {
    opengllogger.log(LogLevel::fatal) << "OpenGLFrameFifo: getAudioFrame: OVERFLOW! No more frames in stack "<<std::endl;
    return NULL;
  }
  tmpframe=stack_audio[0];  // take a pointer to frame from the pre-allocated stack
  stack_audio.pop_front(); // .. remove that pointer from the stack
  return tmpframe;
}
*/


/*
Frame* OpenGLFrameFifo::prepareFrame(Frame* frame) {
  Frame* tmpframe=NULL;
  
#ifdef PRESENT_VERBOSE
  std::cout << "OpenGLFrameFifo: prepareFrame:"<<std::endl;
#endif
  
  tmpframe=getAudioFrame();
  if (!tmpframe) {
    opengllogger.log(LogLevel::normal) << "OpenGLFrameFifo: prepareFrame:  WARNING! Could not get frame from the stack"<<std::endl;
    return NULL;
  }
  *tmpframe=*frame; // deep copy of the frame
#ifdef PRESENT_VERBOSE
  std::cout << "OpenGLFrameFifo: prepareFrame: bye"<<std::endl;
#endif
  
  return tmpframe;
}
*/


Frame* OpenGLFrameFifo::prepareAVFrame(Frame* frame) {// prepare a frame that is about to be inserted into the presentaton infifo
  // Feed here only avframes and video!
  
  /*
  if (frame->frametype!=FrameType::avframe) {
    opengllogger.log(LogLevel::normal) << "OpenGLFrameFifo: prepareAVFrame: WARNING! Frame is not an avframe "<< *frame << std::endl; // << " " << int(frame->frametype) << " " << int(FrameType::avframe) << std::endl;
    return NULL;
  }
  */
  
  Frame* tmpframe    =NULL;
  GLsizei planesize_y =0;
  GLsizei planesize_u =0;
  GLsizei planesize_v =0;
  
  // frames that are pulled from the stacks have their yuvpbo attribute enabled 
  if ( // ALLOWED PIXEL FORMATS // NEW_CODEC_DEV: is your pixel format supported?
    frame->av_codec_context->pix_fmt==  AV_PIX_FMT_YUV420P  ||
    frame->av_codec_context->pix_fmt==  AV_PIX_FMT_YUVJ420P
  ) {
    
    planesize_y=(frame->av_frame->height)  *(frame->av_frame->linesize[0]);
    planesize_u=(frame->av_frame->height/2)*(frame->av_frame->linesize[1]);
    planesize_v=(frame->av_frame->height/2)*(frame->av_frame->linesize[2]);
    
    if      (planesize_y <= BitmapPars::N720::size)  { // frames obtained with getFrame will be recycled by the presentation routine
      tmpframe=getFrame(BitmapPars::N720::type); // handling stacks with getFrame is mutex protected
    }
    else if (planesize_y <= BitmapPars::N1080::size) {
      tmpframe=getFrame(BitmapPars::N1080::type);
    }
    else if (planesize_y <= BitmapPars::N1440::size) {
      tmpframe=getFrame(BitmapPars::N1440::type);
    }
    else if (planesize_y <= BitmapPars::N4K::size)   {
      tmpframe=getFrame(BitmapPars::N4K::type);
    }
    else {
      opengllogger.log(LogLevel::fatal) << "OpenGLFrameFifo: prepareAVFrame:  WARNING! Could not get frame dimensions "<< *frame <<std::endl;
      if (opengllogger.log_level>=LogLevel::normal) { reportStacks(); }
      return NULL;
    }
    
    if (!tmpframe) {
      opengllogger.log(LogLevel::normal) << "OpenGLFrameFifo: prepareAVFrame:  WARNING! Could not get frame from the stack"<<std::endl;
      return NULL;
    }
    
    if (tmpframe->frametype!=FrameType::yuvframe) {
      opengllogger.log(LogLevel::fatal) << "OpenGLFrameFifo: prepareAVFrame:  WARNING! frames in stack are not initialized to FrameType::yuvframe.  Did you forget to start OpenGLThread?"<<std::endl;
      return NULL;
    }
    
    // std::cout << "yuvpbo>"<<tmpframe->yuvpbo<<std::endl;
    
    frame->copyMeta(tmpframe); // timestamps, slots, etc.
    (tmpframe->yuv_pars).pix_fmt =frame->av_codec_context->pix_fmt;
    (tmpframe->yuv_pars).width   =frame->av_frame->linesize[0];
    (tmpframe->yuv_pars).height  =frame->av_frame->height;
    
    // planesize =(frame->av_frame->height)*(frame->av_frame->linesize[0]);
    
#ifdef PRESENT_VERBOSE
    std::cout << "OpenGLFrameFifo: prepareAVFrame:  av_frame->height, av_frame->linesize[0], planesize "<< frame->av_frame->height << " " << frame->av_frame->linesize[0] << " " << planesize <<std::endl;
    std::cout << "OpenGLFrameFifo: prepareAVFrame:  payload: "<< int(frame->av_frame->data[0][0]) << " " << int(frame->av_frame->data[1][0]) << " " << int(frame->av_frame->data[2][0]) << std::endl;
#endif
    
    // std::cout << "yuvpbo->size>"<<tmpframe->yuvpbo->size<<std::endl;
    
    ///*
    tmpframe->yuvpbo->upload(planesize_y,
                            planesize_u,
                            planesize_v,
                            frame->av_frame->data[0],
                            frame->av_frame->data[1],
                            frame->av_frame->data[2]); // up to the GPU! :)
    //*/
                            
    return tmpframe;
  } //ALLOWED PIXEL FORMATS
  else {
    opengllogger.log(LogLevel::normal) << "OpenGLFrameFifo: pixel format "<< frame->av_codec_context->pix_fmt <<" not supported "<<std::endl;
  }
  return NULL;
}


bool OpenGLFrameFifo::writeCopy(Frame* f, bool wait) {
  // One should feed here only:
  // (a) frames of FrameType avframe, i.e. AVFrame bitmaps and with codec_type AVMEDIA_TYPE_VIDEO.  
  //     sound frames should be dealed in some other place (i.e. alsa thread .. that will be implemented someday)
  // (b) frames that represent configurations/commands to the opengl thread.  Will be specified later
  
  Frame* tmpframe=NULL;
  long int dt;
  
  if (f->frametype==FrameType::avframe) { // NEW_CODEC_DEV // when adding new codecs: if the decoder you have implemented, returns Frames with avframe enabled, do some thinking here..
    if (f->av_codec_context->codec_type==AVMEDIA_TYPE_VIDEO) { // only video .. audio should never end up here
      tmpframe=prepareAVFrame(f);
    }
  }
  else if (f->frametype==FrameType::glsetup) { // a generic command to OpenGLThread : to-be-specified
  }
    
  if (!tmpframe) {
    opengllogger.log(LogLevel::debug) << "OpenGLFrameFifo: writeCopy: WARNING! could not stage frame "<< *f <<std::endl;
    return false;
  }
  
#ifdef TIMING_VERBOSE
  dt=(getCurrentMsTimestamp()-tmpframe->mstimestamp);
  if (dt>100) {
    std::cout << "OpenGLFrameFifo: writeCopy : timing : inserting frame " << dt << " ms late" << std::endl;
  }
#endif
  
  if (debug) {
    opengllogger.log(LogLevel::normal) << "OpenGLFrameFifo: writeCopy: DEBUG MODE: recycling frame "<< *tmpframe <<std::endl;
    recycle(tmpframe);
    reportStacks();
    return true;
  }
  
  { // mutex protection from this point on
    std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
      
    this->fifo.push_front(tmpframe);
    ++(this->count);
    
#ifdef PRESENT_VERBOSE
    std::cout << "OpenGLFrameFifo: writeCopy: count=" << this->count << " frame="<<*tmpframe<<std::endl;
#endif
    
#ifdef FIFO_VERBOSE
    std::cout << "OpenGLFrameFifo: writeCopy: count=" << this->count <<std::endl;
#endif
    
    this->condition.notify_one(); // after receiving 
    return true;
  }
  
}


/* // just inherit
Frame* OpenGLFrameFifo::read(unsigned short int mstimeout) {// Pop a frame from the end of the fifo and return the frame to the reservoir stack
  return NULL;
}
*/


void OpenGLFrameFifo::recycle(Frame* f) {// Return Frame f back into the stack.  Update target_size if necessary
  std::unique_lock<std::mutex> lk(this->mutex);
  std::deque<Frame*>* tmpstack;
  
  if (f->frametype==FrameType::yuvframe) {// recycle yuvframe
    switch((f->yuv_pars).bmtype) {
      case (BitmapPars::N720::type): {
        tmpstack=&stack_720p;
        break;
      }
      case (BitmapPars::N1080::type): {
        tmpstack=&stack_1080p;
        break;
      }
      case (BitmapPars::N1440::type): {
        tmpstack=&stack_1440p;
        break;
      }
      case (BitmapPars::N4K::type): {
        tmpstack=&stack_4K;
        break;
      }
      default: {
        opengllogger.log(LogLevel::fatal) << "OpenGLFrameFifo: recycle: FATAL! No such bitmap type "<<(f->yuv_pars).bmtype<<std::endl;
        break;
      }
    }
  }
  else { // recycle opengl command frames: to-be-specified
    std::cout << "OpenGLFrameFifo: recycle: weird frame" << std::endl;
    perror("OpenGLFrameFifo: recycle: weird frame");
  }
  
  /*
  else {// recycle any other type of frame (i.e. audio)
    // opengllogger.log(LogLevel::normal) << "OpenGLFrameFifo: recycle: WARNING! None of the stacks accepts frame "<< *f <<std::endl;
    tmpstack=&stack_audio;
  }
  */
  
#ifdef PRESENT_VERBOSE
  std::cout << "OpenGLFrameFifo: recycle: recycling frame "<<*f<<std::endl;
#endif
  // reportStacks_();
  // f->yuvpbo=NULL; // just in case .. // NOT THIS! the reserved YUVPBO would get lost!
  // std::cout << "1>>>"<<(*tmpstack).size()<<std::endl;
  // (*tmpstack).push_front(f); 
  
  // tmpstack->push_front(f); // pop_front/push_front : use stack first in-first out fashion :  TODO: jitter when using this..!
  tmpstack->push_back(f); // pop_front/push_back : use stack in first in-last out (cyclic) fashion
  
  // std::cout << "2>>>"<<(*tmpstack).size()<<std::endl;
  // reportStacks_();
}


void OpenGLFrameFifo::reportStacks_() {
  std::cout<<"OpenGLFrameFifo reportStacks: "<<std::endl;
  std::cout<<"OpenGLFrameFifo reportStacks: "<<"Stack   Reservoir, Free Stack" <<std::endl;
  std::cout<<"OpenGLFrameFifo reportStacks: "<< "720p    "<<reservoir_720p .size()<<", "<<stack_720p.  size() <<std::endl;
  std::cout<<"OpenGLFrameFifo reportStacks: "<< "1080p   "<<reservoir_1080p.size()<<", "<<stack_1080p.size()  <<std::endl;
  std::cout<<"OpenGLFrameFifo reportStacks: "<< "1440p   "<<reservoir_1440p.size()<<", "<<stack_1440p.size()  <<std::endl;
  std::cout<<"OpenGLFrameFifo reportStacks: "<< "4K      "<<reservoir_4K   .size()<<", "<<stack_4K.   size()  <<std::endl;
  // std::cout<<"OpenGLFrameFifo reportStacks: "<< "audio   "<<reservoir_audio.size()<<", "<<stack_audio.size()  <<std::endl;
  std::cout<<"OpenGLFrameFifo reportStacks: "<<std::endl;
}


void OpenGLFrameFifo::dumpStack_() {
  unsigned short int i=0;
  queuelogger.log(LogLevel::normal) << "OpenGLFrameFifo: dumpStack>" << std::endl;
  queuelogger.log(LogLevel::normal) << "OpenGLFrameFifo: 720p" << std::endl;
  FrameFifo::dump(stack_720p);
  queuelogger.log(LogLevel::normal) << "OpenGLFrameFifo: 1080p" << std::endl;
  FrameFifo::dump(stack_1080p);
  queuelogger.log(LogLevel::normal) << "OpenGLFrameFifo: 1440p" << std::endl;
  FrameFifo::dump(stack_1440p);
  queuelogger.log(LogLevel::normal) << "OpenGLFrameFifo: 4K" << std::endl;
  FrameFifo::dump(stack_4K);
  //queuelogger.log(LogLevel::normal) << "OpenGLFrameFifo: audio" << std::endl;
  //FrameFifo::dump(stack_audio);
  queuelogger.log(LogLevel::normal) << "OpenGLFrameFifo: <dumpStack" << std::endl;
}


void OpenGLFrameFifo::diagnosis_() {
  std::cout << "FIFO: " <<  fifo.size() << " 720p: " << stack_720p.size() << " 1080p: " << stack_1080p.size() << " 1440p: " << stack_1440p.size() << " 4K: " << stack_4K.size() << std::endl; // << " audio: " << stack_audio.size() << std::endl;
}


void OpenGLFrameFifo::reportStacks() {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  reportStacks_();
}


void OpenGLFrameFifo::dumpStack() {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  dumpStack_();
}

void OpenGLFrameFifo::diagnosis() {
  std::unique_lock<std::mutex> lk(this->mutex); // this acquires the lock and releases it once we get out of context
  diagnosis_();
}
  

/*
void OpenGLFrameFifo::checkOrder() {
  std::unique_lock<std::mutex> lk(this->mutex);
  long int mstimestamp=0;
  std::cout << "paska" << std::endl;
  for(auto it=fifo.begin(); it!=fifo.end(); ++it) {
    std::cout << ">>>>" << (*it)->mstimestamp-mstimestamp << std::endl;
    if ((*it)->mstimestamp>mstimestamp) {
      std::cout << "OpenGLFrameFifo : checkOrder : Discontinuity in fifo! :" << (*it)->mstimestamp << " <= " << mstimestamp << std::endl;
    }
    mstimestamp=(*it)->mstimestamp;
  }
}
*/



SlotContext::SlotContext() : yuvtex(NULL), shader(NULL), active(false), codec_id(AV_CODEC_ID_NONE) {
}


SlotContext::~SlotContext() {
 deActivate(); 
}


void SlotContext::activate(GLsizei w, GLsizei h, YUVShader* shader) {//Allocate SlotContext::yuvtex and SlotContext::shader
  if (active) {
    deActivate();
  }
  this->shader=shader;
  opengllogger.log(LogLevel::crazy) << "SlotContext: activate: activating for w, h " << w << " " << h << " " << std::endl;
  yuvtex=new YUVTEX(w, h);
  // shader=new YUVShader(); // nopes..
  active=true;
}


void SlotContext::deActivate() {//Deallocate
  active=false;
  if (yuvtex!=NULL) {delete yuvtex;}
  // if (shader!=NULL) {delete shader;}
  yuvtex=NULL;
  // shader=NULL;
}


void SlotContext::loadTEX(YUVPBO* pbo, long int mstimestamp) {
#ifdef PRESENT_VERBOSE
  std::cout << "SlotContext: loadTEX: pbo: "<< *pbo <<std::endl;
#endif
#ifdef OPENGL_TIMING
  if (mstimestamp<=prev_mstimestamp) { // check that we have fed the frames in correct order (per slot)
    std::cout << "loadTEX: feeding frames in reverse order!" << std::endl;
  }
  prev_mstimestamp=mstimestamp
#endif
  
  loadYUVTEX(pbo, this->yuvtex);
  // this->pbo=pbo; // nopes ..
}

/*
void SlotContext::loadTEX() {
  loadYUVTEX(this->pbo, this->yuvtex);
}
*/
  

// RenderContext::RenderContext(Shader *shader, const YUVTEX &yuvtex, const unsigned int z) : shader(shader), yuvtex(yuvtex), z(z) {
RenderContext::RenderContext(const SlotContext &slot_context, unsigned int z) : slot_context(slot_context), z(z), active(false) {
  // https://learnopengl.com/#!Getting-started/Textures
  // https://www.khronos.org/opengl/wiki/Vertex_Specification
  // https://gamedev.stackexchange.com/questions/86125/what-is-the-relationship-between-glvertexattribpointer-index-and-glsl-location
  // https://stackoverflow.com/questions/5532595/how-do-opengl-texture-coordinates-work
  // So, what is the vertex array object all about:
  // https://stackoverflow.com/questions/8704801/glvertexattribpointer-clarification
  
  /* // bad idea..
  struct timeval time;
  gettimeofday(&time, NULL);
  id=time.tv_sec*1000+time.tv_usec/1000;
  */
  id = std::rand();
  activate();
}


RenderContext::~RenderContext() {
  if (active) {
    glDeleteBuffers(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
  }
}

bool RenderContext::activateIf() {
 if (!active) { // not active .. try to activate
  return activate(); 
 }
 else {
   return true;
 }
}


bool RenderContext::activate() { 
  if (!slot_context.isActive()) {
    return false; // could not activate..
  }
  // so, slot context has been properly initialized => we have Shader and YUVTEX instances
  active=true;
  Shader* shader=slot_context.shader;
  struct timeval time;
  unsigned int transform_size, vertices_size, indices_size;
  
  transform =std::array<GLfloat,16>{
    1.0f,             0.0f,             0.0f,   0.0f, 
    0.0f,             1.0f,             0.0f,   0.0f,
    0.0f,             0.0f,             1.0f,   0.0f,
    0.0f,             0.0f,             0.0f,   1.0f
  };
  transform_size=sizeof(GLfloat)*transform.size();
  
  vertices =std::array<GLfloat,20>{
    /* Positions          Texture Coords
       Shader class references:
       "position"        "texcoord"
    */
    1.0f,  1.0f, 0.0f,   1.0f, 1.0f, // Top Right
    1.0f, -1.0f, 0.0f,   1.0f, 0.0f, // Bottom Right
   -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, // Bottom Left
   -1.0f,  1.0f, 0.0f,   0.0f, 1.0f  // Top Left 
  };
  vertices_size=sizeof(GLfloat)*vertices.size();
  
  indices =std::array<GLuint,6>{  // Note that we start from 0!
    0, 1, 3, // First Triangle
    1, 2, 3  // Second Triangle
  };
  indices_size=sizeof(GLuint)*indices.size();
  
  
  // std::cout << "SIZEOF: " << sizeof(vertices) << " " << vertices_size << std::endl; // eh.. its the same
  
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);
  
  opengllogger.log(LogLevel::crazy) << "RenderContext: activate: VAO, VBO, EBO " << VAO << " " << VBO << " " << EBO << std::endl;
  opengllogger.log(LogLevel::crazy) << "RenderContext: activate: position, texcoord " << shader->position << " " << shader->texcoord << " " << std::endl;
  
  glBindVertexArray(VAO); // VAO works as a "mini program" .. we do all the steps below, when binding the VAO
  
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  // glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices.data(), GL_STATIC_DRAW);
  glBufferData(GL_ARRAY_BUFFER, vertices_size, vertices.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  // glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices.data(), GL_STATIC_DRAW);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_size, indices.data(), GL_STATIC_DRAW);
  
  // Position attribute
  glVertexAttribPointer(shader->position, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0); // this refers to the "Positions" part of vertices
  // 0: shader prog ref, 3: three elements per vertex
  glEnableVertexAttribArray(shader->position); // this refers to (location=0) in the shader program
  // TexCoord attribute
  glVertexAttribPointer(shader->texcoord, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat))); // this refers to "Texture Coords" part of vertices
  // 1: shader prog ref, 2: two elements per vertex
  glEnableVertexAttribArray(shader->texcoord); // this refers to (location=1) in the shader program
  
  glBindVertexArray(0); // Unbind VAO
  
  return true;
}



void RenderContext::render(XWindowAttributes x_window_attr) {// Calls bindTextures, bindParameters and bindVertexArray
  Shader* shader=slot_context.shader;
  
// in the old code:
// shader->use() .. copy from pbo to tex .. makeCurrent .. glViewport, glClear .. bindVertex, drawelements, unbind vertex array
// here:
// handleFifo: loadTex: loadYUVTEX: copy from pbo to tex .. RenderGroup::render: makeCurrent .. glViewport, glClear .. RenderContext::render: shader->use() .. 
  
#ifdef RENDER_VERBOSE
  std::cout << "RenderContext: render: " << std::endl;
#endif
  if (activateIf()) {
#ifdef RENDER_VERBOSE
    std::cout << "RenderContext: render: rendering!" << std::endl;
#endif
    
#ifdef OPENGL_TIMING
    long int mstime = getCurrentMsTimestamp();
    long int swaptime = mstime;
#endif
    
    shader->use(); // use the shader
    this->x_window_attr=x_window_attr;
    
    bindTextures();
    
#ifdef OPENGL_TIMING
  swaptime=mstime; mstime=getCurrentMsTimestamp();
  if ( (mstime-swaptime) > 2 ) {
    std::cout << "RenderContext: render : render timing       : " << mstime-swaptime << std::endl;
  }
#endif
  
    bindVars();
    
#ifdef OPENGL_TIMING
  swaptime=mstime; mstime=getCurrentMsTimestamp();
  if ( (mstime-swaptime) > 2 ) {
    std::cout << "RenderContext: render : bindvars timing     : " << mstime-swaptime << std::endl;
  }
#endif
    
    bindVertexArray();
    
#ifdef OPENGL_TIMING
  swaptime=mstime; mstime=getCurrentMsTimestamp();
  if ( (mstime-swaptime) > 2 ) {
    std::cout << "RenderContext: render : bindvertexarr timing: " << mstime-swaptime << std::endl;
  }
#endif
    
  }
  else {
#ifdef RENDER_VERBOSE
    std::cout << "RenderContext: render: could not render (not active)" << std::endl;
#endif
  }
}


void RenderContext::bindTextures() {// Associate textures with the shader program.  Uses parameters from Shader reference.
  YUVShader *shader = (YUVShader*)(slot_context.shader); // shorthand
  YUVTEX *yuvtex    = (YUVTEX*)   (slot_context.yuvtex);
  
#ifdef RENDER_VERBOSE
  std::cout << "RenderContext: bindTextures: indices y, u, v = " << yuvtex->y_index <<" "<< yuvtex->u_index <<" "<< yuvtex->v_index << std::endl;
  std::cout << "RenderContext: bindTextures: shader refs     = " << shader->texy <<" "<< shader->texu <<" "<< shader->texv << std::endl;
#endif
  
  // slot_context.loadTEX(); // not necessary
  
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, yuvtex->y_index);
  glUniform1i(shader->texy, 0); // pass variable to shader
  
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, yuvtex->u_index);
  glUniform1i(shader->texu, 1); // pass variable to shader
  
  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, yuvtex->v_index);
  glUniform1i(shader->texv, 2); // pass variable to shader
}


void RenderContext::bindVars() {// Upload other data to the GPU (say, transformation matrix).  Uses parameters from Shader reference.
  // eh.. now we're doing this on each "sweep" .. we could give a callback to RenderGroup that would do the uploading of transformatio matrix
  // .. on the other hand, its not that much data (compared to bitmaps) !
  YUVShader *shader = (YUVShader*)(slot_context.shader); // shorthand
  YUVTEX *yuvtex    = (YUVTEX*)(slot_context.yuvtex);
  
  XWindowAttributes& wa=x_window_attr; // shorthand
  GLfloat r, dx, dy;
    
  // calculate dimensions
  // (screeny/screenx) / (iy/ix)  =  screeny*ix / screenx*iy
  r=float(wa.height*(yuvtex->w)) / float(wa.width*(yuvtex->h));
  if (r<1.){ // screen wider than image
    dy=1;
    dx=r;
  }
  else if (r>1.) { // screen taller than image
    dx=1;
    dy=1/r;
  }
  else {
    dx=1;
    dy=1;
  }
  
#ifdef RENDER_VERBOSE
  std::cout << "RenderContext: bindVars: dx, dy = " << dx <<" "<<dy<<" "<< std::endl;
#endif  
  // /* // test..
  transform[0]=dx;
  transform[5]=dy;  
  // */
  /*
  transform= {
    {dx                0.0f,             0.0f,   0.0f}, 
    {0.0f,             dy                0.0f,   0.0f},
    {0.0f,             0.0f,             1.0f,   0.0f},
    {0.0f,             0.0f,             0.0f,   1.0f}
  };
  */
  glUniformMatrix4fv(shader->transform, 1, GL_FALSE, transform.data());
}


void RenderContext::bindVertexArray() {// Bind the vertex array and draw
  
#ifdef RENDER_VERBOSE
  std::cout << "RenderContext: bindVertexArray: VAO= " << VAO << std::endl;
#endif
  
#ifdef TIMING_VERBOSE
  long int dt=getCurrentMsTimestamp();
#endif
  
  glBindVertexArray(VAO);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);
  
#ifdef TIMING_VERBOSE
  dt=getCurrentMsTimestamp()-dt;
  if (dt>10) {
    std::cout << "RenderContext: bindVertexArray : timing : drawing took " << dt << " ms" <<std::endl;
  }
#endif
  
  
#ifdef RENDER_VERBOSE
  std::cout << "RenderContext: bindVertexArray: " << std::endl;
#endif
  
}

/*
void RenderContext::unBind() {
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0); // unbind
  glBindTexture(GL_TEXTURE_2D, 0); // unbind 
}
*/



RenderGroup::RenderGroup(Display* display_id, const GLXContext& glc, Window window_id, Window child_id, bool doublebuffer_flag) : display_id(display_id), glc(glc), window_id(window_id), child_id(child_id), doublebuffer_flag(doublebuffer_flag) {
}


RenderGroup::~RenderGroup() {
}


std::list<RenderContext>::iterator RenderGroup::getContext(int id) {
  std::list<RenderContext>::iterator it;
  for(it=render_contexes.begin(); it!=render_contexes.end(); ++it) {
    if (it->getId()==id) {
      return it;
    }
  }
  return it;
}


bool RenderGroup::addContext(RenderContext render_context) {
  if (getContext(render_context.getId())==render_contexes.end()) {
    render_contexes.push_back(render_context); 
    return true;
  }
  else { // there is a RenderContext with the same id here..
    return false;
  }
}


bool RenderGroup::delContext(int id) {
  std::list<RenderContext>::iterator it = getContext(id);
  if (it==render_contexes.end()) {
    return false;
  }
  else {
    render_contexes.erase(it); // this drives the compiler mad.. it starts moving stuff in the container..?
    // render_contexes.pop_back(); // this is ok..
    return true;
  }
}


bool RenderGroup::isEmpty() {
  return render_contexes.empty();
}


void RenderGroup::render() {
  
#ifdef PRESENT_VERBOSE
  std::cout << "RenderGroup: " << std::endl;
  std::cout << "RenderGroup: start rendering!" << std::endl;
  std::cout << "RenderGroup: render: display, window_id, child_id" <<display_id<<" "<<window_id<<" "<<child_id << std::endl;
#endif
  
  // glFlush();
  // glFinish();
  
#ifdef OPENGL_TIMING
  long int mstime =getCurrentMsTimestamp();
  long int swaptime;
#endif
  
  if (!glXMakeCurrent(display_id, child_id, glc)) { // choose this x window for manipulation
    opengllogger.log(LogLevel::normal) << "RenderGroup: render: WARNING! could not draw"<<std::endl;
  }
  XGetWindowAttributes(display_id, child_id, &(x_window_attr));
  
#ifdef OPENGL_TIMING
  swaptime=mstime; mstime=getCurrentMsTimestamp();
  if ( (mstime-swaptime) > 2 ) {
    std::cout << "RenderGroup: render : gxlmakecurrent timing : " << mstime-swaptime << std::endl;
  }
#endif
  
#ifdef PRESENT_VERBOSE
  std::cout << "RenderGroup: render: window w, h " <<x_window_attr.width<<" "<<x_window_attr.height<<std::endl;
#endif
  
  glFinish(); // TODO: debugging
  
  glViewport(0, 0, x_window_attr.width, x_window_attr.height);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // clear the screen and the depth buffer
  
  for(std::list<RenderContext>::iterator it=render_contexes.begin(); it!=render_contexes.end(); ++it) {
    it->render(x_window_attr);
  }
  
#ifdef OPENGL_TIMING
  swaptime=mstime; mstime=getCurrentMsTimestamp();
  if ( (mstime-swaptime) > 2 ) {
    std::cout << "RenderGroup: render : render timing         : " << mstime-swaptime << std::endl;
  }
#endif
  
  
  if (doublebuffer_flag) {
#ifdef PRESENT_VERBOSE
    std::cout << "RenderGroup: render: swapping buffers "<<std::endl;
#endif
    glXSwapBuffers(display_id, child_id);
  }
  
  // glFlush();
  // glFinish();
  
#ifdef OPENGL_TIMING
  swaptime=mstime; mstime=getCurrentMsTimestamp();
  if ( (mstime-swaptime) > 2 ) {
    std::cout << "RenderGroup: render : swap buffer timing    : " << mstime-swaptime << std::endl;
  }
#endif

  // glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0); // unbind
  // glBindTexture(GL_TEXTURE_2D, 0); // unbind
  // glFinish(); // TODO: necessary?
  
#ifdef PRESENT_VERBOSE
  std::cout << "RenderGroup: render: stop rendering!"<<std::endl;
#endif
}



OpenGLThread::OpenGLThread(const char* name, unsigned short n720p, unsigned short n1080p, unsigned short n1440p, unsigned short n4K, unsigned msbuftime, int core_id) : Thread(name, core_id), infifo(n720p, n1080p, n1440p, n4K), msbuftime(msbuftime), debug(false) {
  // So, what happens here..?
  // We create the OpenGLFrameFifo instance "infifo" at constructor time, and then pass "infifo" to AVFifoFrameFilter instance "framefilter" as a parameter
  // * framefilter (AVFifoFrameFilter) knows infifo
  // * infifo (OpenGLFrameFifo) has series of stacks 
  // * The stacks are initialized by OpenGLThread at OpenGLThread::preRun() => OpenGLThread::reserveFrames()
  // * .. there the Frame instances of stacks get their yuvpbo members initialized
  
  int i;
  for(i=0;i<=I_MAX_SLOTS;i++) {
    slots_.push_back(SlotContext());
    // std::cout << ">>"<<slots_.back().active<< " " << slots_[0].active << std::endl;
  }

  for(i=0;i<=I_MAX_SLOTS;i++) {
    std::list<RenderGroup*> lis;
    render_lists.push_back(lis);
  }
  
  std::srand(std::time(0)); // init random number generator
  
  if (msbuftime<100) {
    opengllogger.log(LogLevel::normal) << "OpenGLThread: constructor: WARNING: your buffering time is very small! Only "<<msbuftime<< " milliseconds: lots of frames might be scrapped"<<std::endl;
  }
  
  future_ms_tolerance=msbuftime*5; // frames this much in the future will be scrapped
  
  slot_times.resize(I_MAX_SLOTS+1,0);
  
  resetCallTime();
}


OpenGLThread::~OpenGLThread() {// WARNING: deallocations shoud be in postRun, i.e. before thread join
  opengllogger.log(LogLevel::debug) << "OpenGLThread: destructor "<<std::endl;
}


unsigned OpenGLThread::getSwapInterval(GLXDrawable drawable) {
  unsigned i;
  if (drawable==0) {
    drawable=root_id;
  }
  switch (swap_flavor) {
    case swap_flavors::ext: {
      glXQueryDrawable(display_id, drawable, GLX_SWAP_INTERVAL_EXT, &i);
      break;
    }
    case swap_flavors::mesa: {
      i=(*pglXGetSwapIntervalMESA)();
      break;
    }
    default: {
      opengllogger.log(LogLevel::normal) << "OpenGLThread::setSwapInterval: could not set swap interval" << std::endl;
      break;
    }
  }
  return i;
}
  
void OpenGLThread::setSwapInterval(unsigned i, GLXDrawable drawable) {
  if (drawable==0) {
    drawable=root_id;
  }
  switch (swap_flavor) {
    case swap_flavors::ext: {
      // I could load this with glxext.h, but including that creates a crash
      // glXSwapIntervalEXT(display_id, drawable, i); // this is annoying .. for "GL_EXT_swap_control" we specify a drawable, for the mesa version, we don't..
      (*pglXSwapIntervalEXT)(display_id, drawable, i);
      break;
    }
    case swap_flavors::mesa: {
      (*pglXSwapIntervalMESA)(i);
      break;
    }
    default: {
      opengllogger.log(LogLevel::normal) << "OpenGLThread::setSwapInterval: could not set swap interval" << std::endl;
      break;
    }
  }
}


bool OpenGLThread::hasRenderGroup(Window window_id) {
  auto search = render_groups.find(window_id);
  
  if (search==render_groups.end()) {
    return false;
  } 
  else {
    return true;
  }
}


RenderGroup& OpenGLThread::getRenderGroup(Window window_id) {// Returns a reference to RenderGroup in OpenGLThread::render_groups
  return render_groups.at(window_id);
}


bool OpenGLThread::newRenderGroup(Window window_id) {
  if (hasRenderGroup(window_id)) {
    return false;
  }
  Window child_id;
  
  reConfigWindow(window_id);
  child_id=window_id;
  
  // child_id =getChildWindow(window_id); // X11 does not create nested windowses .. that's a job for the window manager, right?
  
  RenderGroup rg(display_id, glc, window_id, child_id, doublebuffer_flag); // window_id = RenderGroup index, child_id = actual window id
  render_groups.insert(std::pair<Window,RenderGroup>(window_id,rg));
  return true;
}


bool OpenGLThread::delRenderGroup(Window window_id) {
  if (!hasRenderGroup(window_id)) {
    opengllogger.log(LogLevel::debug) << "OpenGLThread: delRenderGroup: no such group "<< window_id << std::endl;
    return false;
  }
  opengllogger.log(LogLevel::debug) << "OpenGLThread: delRenderGroup: erasing group "<< window_id << std::endl;
  
  // https://stackoverflow.com/questions/596162/can-you-remove-elements-from-a-stdlist-while-iterating-through-it
  // render_list: vector of lists, each vector element (and list) corresponds to a certain slot
  for(auto it=render_lists.begin(); it!=render_lists.end(); ++it) { // gives each list
    auto it2=it->begin(); // *(it) = list of RenderGroup(s)
    while(it2 !=it->end()) { // *(it2) = RenderGroup instance
      if ( (*it2)->getWindowId() == window_id ) {
        opengllogger.log(LogLevel::debug) << "OpenGLThread: delRenderGroup: erasing reference to "<< window_id << std::endl;
        it2=it->erase(it2);
      }
      else {
        ++it2;
      }
    }
  }
  
  render_groups.erase(render_groups.find(window_id));
  return true;
}


int OpenGLThread::newRenderContext(SlotNumber n_slot, Window window_id, unsigned int z) {
  if (!slotOk(n_slot)) {return 0;}
  
  const SlotContext& slot_context=slots_[n_slot]; // take a reference to the relevant SlotContext
  
  if (hasRenderGroup(window_id)) {
    RenderGroup* render_group= &render_groups.at(window_id); // reference to a RenderGroup corresponding to window_id
    RenderContext render_context(slot_context,z); // the brand new RenderContext
    render_group->addContext(render_context);
    
    render_lists[n_slot].push_back(render_group); // render_lists[n_slot] == list of RenderGroup instances
    // n_slot => render_lists[n_slot] => a list of RenderGroups, i.e. x windowses that the stream of this slot is streaming into    
    opengllogger.log(LogLevel::debug) << "OpenGLThread: newRenderContext: added new RenderContext" << std::endl;
    opengllogger.log(LogLevel::debug) << "OpenGLThread: newRenderContext: render_list at slot " << n_slot << " now with size " << render_lists[n_slot].size() << std::endl;
    // render_lists[1].size()
    
    // reportSlots();
    
    return render_context.getId();
  }
  else {
    return 0;
  }
}


bool OpenGLThread::delRenderContext(int id) {
  bool removed=false;
  
  for(std::map<Window,RenderGroup>::iterator it=render_groups.begin(); it!=render_groups.end(); ++it) {
    if ((it->second).delContext(id)) {
      opengllogger.log(LogLevel::debug) << "OpenGLThread: delRenderContext: removed context " << id << std::endl;
      removed=true;
    }
  }
  
  // remove empty render groups from render list
  for(auto it=render_lists.begin(); it!=render_lists.end(); ++it) { // *(it) == std::list
    auto it2=it->begin(); // *(it2) == *RenderGroup
    while(it2 != it->end()) {
      if ((*it2)->isEmpty()) {// remove from list
        it2=it->erase(it2); // remove *(it2) [RenderGroup] from *(it1) [std::list]
      }
      else {
        ++it2;
      }
    }
  }

return removed;
}


void OpenGLThread::delRenderContexes() {
  // for each render group, empty the render_contexes list
  for(std::map<Window,RenderGroup>::iterator it=render_groups.begin(); it!=render_groups.end(); ++it) {
    (it->second).render_contexes.erase((it->second).render_contexes.begin(),(it->second).render_contexes.end());
  }
}

   

void OpenGLThread::loadTEX(SlotNumber n_slot, YUVPBO* pbo, long int mstimestamp){// Load PBO to texture in slot n_slot
  // if (!slotOk(n_slot)) {return 0;} // assume checked (this is for internal use only)
  slots_[n_slot].loadTEX(pbo, mstimestamp);
}
  

OpenGLFrameFifo& OpenGLThread::getFifo() {
  return infifo;
}


void OpenGLThread::activateSlot(SlotNumber i, YUVFramePars yuv_pars) {
  GLsizei w, h;
  opengllogger.log(LogLevel::crazy) << "OpenGLThread: activateSlot: "<<yuv_pars.bmtype<<" "<<yuv_pars.width<< " "<< yuv_pars.height << std::endl;
  slots_[i].activate(yuv_pars.width, yuv_pars.height, yuv_shader);
}


void OpenGLThread::activateSlotIf(SlotNumber i, YUVFramePars yuv_pars) {
  if (slots_[i].isActive()) {
    if (yuv_pars.width==slots_[i].yuvtex->w and yuv_pars.height==slots_[i].yuvtex->h) { // nothings changed ..
    }
    else {
      opengllogger.log(LogLevel::debug) << "OpenGLThread: activateSlotIf: texture dimensions changed: reactivate" << std::endl;
      activateSlot(i, yuv_pars);
    }
  } 
  else { // not activated => activate
    activateSlot(i, yuv_pars);
  }
}


bool OpenGLThread::slotTimingOk(SlotNumber n_slot, long int mstime) {
  // std::cout << "slotTimingOk: " << mstime-slot_times[n_slot] << std::endl;
  
  if ( (mstime-slot_times[n_slot]<=10) ) { // too little time passed .
    slot_times[n_slot]=mstime;
    return false;
  }
  else {
    slot_times[n_slot]=mstime;
    return true;
  }
}
  

void OpenGLThread::render(SlotNumber n_slot) {// Render all RenderGroup(s) depending on slot n_slot
  // if (!slotOk(n_slot)) {return;} // assume checked
  // if (slots_[n_slot].isActive()) {// assume active
    for (auto it=render_lists[n_slot].begin(); it!=render_lists[n_slot].end(); ++it) { // iterator has the RenderGroup*
      (*it)->render();
    }
  //}
}


void OpenGLThread::dumpFifo() {
  long int mstime=getCurrentMsTimestamp();
  long int rel_mstime;
  long int mstimestamp=9516360576679;
  
  // std::cout<<std::endl<<"OpenGLThread: dumpfifo: "<< mstime-msbuftime <<std::endl;
  for(auto it=presfifo.begin(); it!=presfifo.end(); ++it) {
    
    if (mstimestamp<((*it)->mstimestamp)) { // next smaller than previous.  Should be: [young (big value) .... old (small value)]
      std::cout<<"OpenGLThread: dumpfifo: JUMP!" << std::endl;
    }
    mstimestamp=(*it)->mstimestamp;
    
    rel_mstime=(*it)->mstimestamp-(mstime-msbuftime);
    // std::cout<<"OpenGLThread: dumpfifo: "<<**it<<" : "<< rel_mstime <<std::endl;
    // std::cout<<"OpenGLThread: dumpfifo: "<< rel_mstime <<std::endl;
    std::cout<<"OpenGLThread: dumpfifo: "<< rel_mstime << " <" << mstimestamp << "> " << std::endl;
  }
  std::cout<<"OpenGLThread: dumpfifo: "<<std::endl;
}


void OpenGLThread::diagnosis() {
  infifo.diagnosis();
  std::cout << "PRESFIFO: " << presfifo.size() << std::endl;
}


void OpenGLThread::resetCallTime() {
  callswaptime =0;
  calltime     =getCurrentMsTimestamp();
  // std::cout << "OpenGLThread: resetCallTime  : " << std::endl;
}


void OpenGLThread::reportCallTime(unsigned i) {
  callswaptime =calltime;
  calltime     =getCurrentMsTimestamp();
  std::cout << "OpenGLThread: reportCallTime : ("<<i<<") "<< calltime-callswaptime << std::endl;
}


long unsigned OpenGLThread::insertFifo(Frame* f) {// sorted insert
  /*
  timestamps in the presentation queue/fifo:
  
  <young                         old>
  90  80  70  60  50  40  30  20  10
  
  see also the comments in OpenGLThread::handleFifo()
  */
  bool inserted=false;
  long int rel_mstimestamp;
  
  auto it=presfifo.begin();
  while(it!=presfifo.end()) {
#ifdef PRESENT_VERBOSE // the presentation fifo will be extremely busy.. avoid unnecessary logging
    // std::cout << "OpenGLThread: insertFifo: iter: " << **it <<std::endl;
#endif
    if (f->mstimestamp >= (*it)->mstimestamp ) {//insert before this element
#ifdef PRESENT_VERBOSE
      //std::cout << "OpenGLThread: insertFifo: inserting "<< *f <<std::endl; // <<" before "<< **it <<std::endl;
#endif
      presfifo.insert(it,f);
      inserted=true;
      break;
   }
   ++it;
  }
  if (!inserted) {
#ifdef PRESENT_VERBOSE
    // std::cout << "OpenGLThread: insertFifo: inserting "<< *f <<" at the end"<<std::endl;
#endif
    presfifo.push_back(f);
  }
  
  rel_mstimestamp=( presfifo.back()->mstimestamp-(getCurrentMsTimestamp()-msbuftime) );
  rel_mstimestamp=std::max((long int)0,rel_mstimestamp);
  
  if (rel_mstimestamp>future_ms_tolerance) { // fifo might get filled up with frames too much in the future (typically wrong timestamps..) process them immediately
#ifdef PRESENT_VERBOSE
    std::cout << "OpenGLThread: insertFifo: frame in distant future: "<< rel_mstimestamp <<std::endl;
#endif
    rel_mstimestamp=0;
  }
    
#ifdef PRESENT_VERBOSE
    std::cout << "OpenGLThread: insertFifo: returning timeout "<< rel_mstimestamp <<std::endl;
#endif
  return (long unsigned)rel_mstimestamp; //return timeout to next frame
}


long unsigned OpenGLThread::handleFifo() {// handles the presentation fifo
  // Check out the docs for the timestamp naming conventions, etc. in \ref timing
  long unsigned mstime_delta;         // == delta
  long int      rel_mstimestamp;      // == trel = t_ - (t-tb) = t_ - delta
  Frame*        f;                    // f->mstimestamp == t_
  bool          present_frame; 
  long int      mstime;  
  
  // mstime_delta=getCurrentMsTimestamp()-msbuftime; // delta = (t-tb)
  // mstime       =getCurrentMsTimestamp();
  // mstime_delta =mstime-msbuftime;
  
#ifdef TIMING_VERBOSE
  resetCallTime();
#endif
  
  auto it=presfifo.rbegin(); // reverse iterator
  
  while(it!=presfifo.rend()) {// while
    // mstime_delta=getCurrentMsTimestamp()-msbuftime; // delta = (t-tb)
    mstime       =getCurrentMsTimestamp();
    mstime_delta =mstime-msbuftime;
    
    f=*it; // f==pointer to frame
    rel_mstimestamp=(f->mstimestamp-mstime_delta); // == trel = t_ - delta
    if (rel_mstimestamp>0 and rel_mstimestamp<=future_ms_tolerance) {// frames from [inf,0) are left in the fifo
      // ++it;
      break; // in fact, just break the while loop (frames are in time order)
    }
    else {// remove the frame *f from the fifo.  Either scrap or present it
      // 40 20 -20 => found -20
      ++it; // go one backwards => 20 
      it= std::list<Frame*>::reverse_iterator(presfifo.erase(it.base())); // eh.. it.base() gives the next iterator (in forward sense?).. we'll remove that .. create a new iterator on the modded
      // it.base : takes -20
      // erase   : removes -20 .. returns 20
      // .. create a new reverse iterator from 20
      present_frame=false;
      
#ifdef NO_LATE_DROP_DEBUG // present also the late frames
      // std::cout << "OpenGLThread: rel_mstimestamp, future_ms_tolerance : " << rel_mstimestamp << " " << future_ms_tolerance << std::endl;
      if (rel_mstimestamp>future_ms_tolerance) { // fifo might get filled up with future frames .. if they're too much in the future, scrap them
        opengllogger.log(LogLevel::normal) << "OpenGLThread: handleFifo: DISCARDING a frame too far in the future " << rel_mstimestamp << " " << *f << std::endl;
      } 
      else { // .. in all other cases, just present the frame
        present_frame=true;
      }
      
#else
      if (rel_mstimestamp<=-10) {// scrap frames from [-10,-inf)
        // opengllogger.log(LogLevel::normal) << "OpenGLThread: handleFifo: DISCARDING late frame " << " " << rel_mstimestamp << " " << *f << std::endl;
        opengllogger.log(LogLevel::normal) << "OpenGLThread: handleFifo: DISCARDING late frame " << " " << rel_mstimestamp << " <" << f->mstimestamp <<"> " << std::endl;
      }
      else if (rel_mstimestamp>future_ms_tolerance) { // fifo might get filled up with future frames .. if they're too much in the future, scrap them
        opengllogger.log(LogLevel::normal) << "OpenGLThread: handleFifo: DISCARDING a frame too far in the future " << rel_mstimestamp << " " << *f << std::endl;
      }
      else if (rel_mstimestamp<=0) {// present frames from [0,-10)
        present_frame=true;
      }
#endif

      if (present_frame) { // present_frame
        if (!slotOk(f->n_slot)) {//slot overflow, do nothing
        }
        else if (f->frametype==FrameType::yuvframe) {// accepted frametype
#if defined(PRESENT_VERBOSE) || defined(TIMING_VERBOSE)
          // std::cout<<"OpenGLThread: handleFifo: PRESENTING " << *f << std::endl;
          std::cout<<"OpenGLThread: handleFifo: PRESENTING " << rel_mstimestamp << " <"<< f->mstimestamp <<"> " << std::endl;
          if (it!=presfifo.rend()) {
            std::cout<<"OpenGLThread: handleFifo: NEXT       " << (*it)->mstimestamp-mstime_delta << " <"<< (*it)->mstimestamp <<"> " << std::endl;
          }
#endif
          // if next frame was give too fast, scrap it
          if (slotTimingOk(f->n_slot,mstime)) {
            // activateSlotIf(f->n_slot, (f->yuv_pars).bmtype); // activate if not already active
            activateSlotIf(f->n_slot, f->yuv_pars); // activate if not already active
#ifdef TIMING_VERBOSE
            reportCallTime(0);
#endif
            // f->yuv_pbo [YUVPBO] has already been uploaded to GPU.  Now it is loaded to the textures.
            // loadTEX uses slots_[], where each vector element is a SlotContext (=set of textures, a shader program)
            loadTEX(f->n_slot, f->yuvpbo, f->mstimestamp); // timestamp is used only for debugging purposes ..
#ifdef TIMING_VERBOSE
            reportCallTime(1);
#endif
            render(f->n_slot); // renders all render groups that depend on this slot.  A slot => RenderGroups (x window) => list of RenderContext => SlotContext (textures)
#ifdef TIMING_VERBOSE
            reportCallTime(2);
#endif
          }
          else {
            opengllogger.log(LogLevel::normal) << "OpenGLThread: handleFifo: feeding frames too fast! dropping.." << std::endl;
          }
        } // accepted frametype
      }// present frame
      infifo.recycle(f); // codec found or not, frame always recycled
    }// present or scrap
  }// while
  
  if (presfifo.empty()) {
#ifdef PRESENT_VERBOSE
    std::cout<<"OpenGLThread: handleFifo: empty! returning default " << Timeouts::openglthread << " ms timeout " << std::endl;
#endif
    return Timeouts::openglthread;
  }
  else {
    f=presfifo.back();
    mstime_delta=getCurrentMsTimestamp()-msbuftime; // delta = (t-tb)
    rel_mstimestamp=f->mstimestamp-mstime_delta; // == trel = t_ - delta
    rel_mstimestamp=std::max((long int)0,rel_mstimestamp);
#ifdef PRESENT_VERBOSE
    std::cout<<"OpenGLThread: handleFifo: next frame: " << *f <<std::endl;
    std::cout<<"OpenGLThread: handleFifo: timeout   : " << rel_mstimestamp <<std::endl;
#endif
    return (long unsigned)rel_mstimestamp; // time delay until the next presentation event..
  }
}
    
  
void OpenGLThread::run() {// Main execution loop
  // time_t timer;
  // time_t oldtimer;
  long int mstime;
  long int old_mstime;
  
  Frame* f;
  long unsigned timeout;
  
  // time(&timer);
  // oldtimer=timer;
  mstime    =getCurrentMsTimestamp();
  old_mstime=mstime;
  
  loop=true;
  
  timeout=Timeouts::openglthread;
  while(loop) {
#ifdef PRESENT_VERBOSE
    std::cout << "OpenGLThread: "<< this->name <<" : run : timeout = " << timeout << std::endl;
    // std::cout << "OpenGLThread: "<< this->name <<" : run : dumping fifo " << std::endl;
    // infifo.dumpFifo();
    // infifo.reportStacks(); 
    // std::cout << "OpenGLThread: "<< this->name <<" : run : dumping fifo " << std::endl;
#endif
    // std::cout << "OpenGLThread: run : read with timeout : " << timeout << std::endl;
    f=infifo.read(timeout);
    if (!f) { // TIMEOUT : either one seconds has passed, or it's about time to present the next frame..
      if (debug) {
      }
      else {
        timeout=std::min(handleFifo(),Timeouts::openglthread); // present/discard frames and return timeout to the last frame.  Recycles frames.  Returns the timeout
        // std::cout << "OpenGLThread: run : no frame : timeout : " << timeout << std::endl;
      }
    } // TIMEOUT
    else { // GOT FRAME // remember to apply infifo.recycle
      if (debug) {
        opengllogger.log(LogLevel::normal) << "OpenGLThread: "<< this->name <<" : run : DEBUG MODE! recycling received frame "<< *f << std::endl;
        infifo.recycle(f);
        infifo.reportStacks();
      }
      else {
        timeout=std::min(insertFifo(f),Timeouts::openglthread); // insert frame into the presentation fifo, get timeout to the last frame in the presentation fifo
        // ..frame is now in the presentation fifo.. remember to recycle on handleFifo
        while (timeout==0) {
          timeout=std::min(handleFifo(),Timeouts::openglthread);
        }
          
#if defined(PRESENT_VERBOSE) || defined(TIMING_VERBOSE)
        dumpFifo();
        std::cout << "OpenGLThread: " << this->name <<" : run : got frame : timeout : " <<timeout<<std::endl<<std::endl;
#endif
      }
    } // GOT FRAME
    
    // time(&timer);
    mstime=getCurrentMsTimestamp();
    
    // if (difftime(timer,oldtimer)>=1) { // time to check the signals..
    if ( (mstime-old_mstime)>=Timeouts::openglthread ) {
      handleSignals();
      // oldtimer=timer;
      old_mstime=mstime;
#ifdef FIFO_DIAGNOSIS
      diagnosis();
#endif
    }
  }
}


void OpenGLThread::preRun() {// Called before entering the main execution loop, but after creating the thread
  initGLX();
  loadExtensions();
  swap_interval_at_startup=getSwapInterval(); // save the value of vsync at startup
  VSyncOff();
  makeShaders();
  reserveFrames(); // calls glFinish
}


void OpenGLThread::postRun() {// Called after the main execution loop exits, but before joining the thread
  delRenderContexes();
  for(std::vector<SlotContext>::iterator it=slots_.begin(); it!=slots_.end(); ++it) {
    it->deActivate(); // deletes textures
  }
  releaseFrames(); // calls glFinish
  delShaders();
  closeGLX();
}


void OpenGLThread::handleSignals() {
  std::unique_lock<std::mutex> lk(this->mutex);
  // AVConnectionContext connection_ctx;
  int i;
  bool ok;
  
  opengllogger.log(LogLevel::crazy) << "OpenGLThread: handleSignals: " << std::endl;
  
  // if (signal_fifo.empty()) {return;}
  
  // handle pending signals from the signals fifo
  for (auto it = signal_fifo.begin(); it != signal_fifo.end(); ++it) { // it == pointer to the actual object (struct SignalContext)
    /* 
    *(it) == SignalContext: members: signal, *ctx
    SlotNumber    n_slot;        // in: new_render_context
    Window        x_window_id;   // in: new_render_context, new_render_group, del_render_group
    unsigned int  z;             // in: new_render_context
    int           render_ctx;    // in: del_render_context, out: new_render_context
    */
    
    // opengllogger.log(LogLevel::debug) << "OpenGLThread: handleSignals: signal->ctx="<< *(it->ctx) << std::endl; // Signals::exit does not need it->ctx ..
    
    switch (it->signal) {
      case Signals::exit:
        opengllogger.log(LogLevel::debug) << "OpenGLThread: handleSignals: exit" << std::endl;
        loop=false;
        break;
        
      case Signals::info:
        reportRenderGroups();
        reportRenderList();
        break;
        
      case Signals::new_render_group:     // newRenderCroupCall
        opengllogger.log(LogLevel::debug) << "OpenGLThread: handleSignals: new_render_group: "<< *(it->ctx) << std::endl;
        ok=newRenderGroup( (it->ctx)->x_window_id ); // Creates a RenderGroup and inserts it into OpenGLThread::render_groups
        if (!ok) {
          opengllogger.log(LogLevel::normal) << "OpenGLThread: handleSignals: could not create render group for window id " << (it->ctx)->x_window_id << std::endl;
        } 
        else {
          (it->ctx)->success=true;
        }
        break;
        
      case Signals::del_render_group:     // delRenderGroupCall
        opengllogger.log(LogLevel::debug) << "OpenGLThread: handleSignals: del_render_group" << std::endl;
        ok=delRenderGroup( (it->ctx)->x_window_id ); // Remove RenderGroup from OpenGLThread::render_groups
        if (!ok) {
          opengllogger.log(LogLevel::normal) << "OpenGLThread: handleSignals: could not delete render group for window id " << (it->ctx)->x_window_id << std::endl;
        }
        else {
          (it->ctx)->success=true;
        }
        break;
        
      case Signals::new_render_context:   // newRenderContextCall
        opengllogger.log(LogLevel::debug) << "OpenGLThread: handleSignals: new_render_context" << std::endl;
        // (it->ctx)->render_ctx=9; // testing
        i=newRenderContext( (it->ctx)->n_slot, (it->ctx)->x_window_id, (it->ctx)->z);
        if (i==0) {
          opengllogger.log(LogLevel::normal) << "OpenGLThread: handleSignals: could not create render context: slot, x window " << (it->ctx)->n_slot << ", " << (it->ctx)->x_window_id << std::endl;
        }
        (it->ctx)->render_ctx=i; // return value
        break;
        
      case Signals::del_render_context:   // delRenderContextCall
        opengllogger.log(LogLevel::debug) << "OpenGLThread: handleSignals: del_render_context" << std::endl;
        ok=delRenderContext((it->ctx)->render_ctx);
        if (!ok) {
          opengllogger.log(LogLevel::normal) << "OpenGLThread: handleSignals: could not delete render context " << (it->ctx)->render_ctx << std::endl;
        }
        else {
          (it->ctx)->success=true;
        }
        break;
      }
  }
    
  signal_fifo.clear();
  condition.notify_one();
}


void OpenGLThread::sendSignal(SignalContext signal_ctx) {
  std::unique_lock<std::mutex> lk(this->mutex);
  this->signal_fifo.push_back(signal_ctx);  
}


void OpenGLThread::sendSignalAndWait(SignalContext signal_ctx) {
  std::unique_lock<std::mutex> lk(this->mutex);
  this->signal_fifo.push_back(signal_ctx);  
  while (!this->signal_fifo.empty()) {
    this->condition.wait(lk);
  }
}


void OpenGLThread::reserveFrames() {
  for(auto it=infifo.reservoir_720p.begin(); it!=infifo.reservoir_720p.end(); ++it) {
    opengllogger.log(LogLevel::crazy) << "OpenGLThread: reserveFrames: reserving YUVPBO with size "<< BitmapPars::N720::size << std::endl;
    it->yuvpbo     =new YUVPBO(BitmapPars::N720::type);
    it->frametype  =FrameType::yuvframe;
    it->yuv_pars   ={BitmapPars::N720::type};
  }
  // dumpStack();
  // return;
  for(auto it=infifo.reservoir_1080p.begin(); it!=infifo.reservoir_1080p.end(); ++it) {
    it->yuvpbo     =new YUVPBO(BitmapPars::N1080::type);
    it->frametype  =FrameType::yuvframe;
    it->yuv_pars   ={BitmapPars::N1080::type};
  }
  for(auto it=infifo.reservoir_1440p.begin(); it!=infifo.reservoir_1440p.end(); ++it) {
    it->yuvpbo     =new YUVPBO(BitmapPars::N1440::type);
    it->frametype  =FrameType::yuvframe;
    it->yuv_pars   ={BitmapPars::N1440::type};
  }
  for(auto it=infifo.reservoir_4K.begin(); it!=infifo.reservoir_4K.end(); ++it) {
    it->yuvpbo     =new YUVPBO(BitmapPars::N4K::type);
    it->frametype  =FrameType::yuvframe;
    it->yuv_pars   ={BitmapPars::N4K::type};
  }
  glFinish();
  
  /*
  for(auto it=infifo.reservoir_audio.begin(); it!=infifo.reservoir_audio.end(); ++it) {
    it->reserve(DEFAULT_PAYLOAD_SIZE_PCMU);
  }
  */
}


void OpenGLThread::releaseFrames() {
  for(auto it=infifo.reservoir_720p.begin(); it!=infifo.reservoir_720p.end(); ++it) {
    delete it->yuvpbo;
    it->reset();
  }
  for(auto it=infifo.reservoir_1080p.begin(); it!=infifo.reservoir_1080p.end(); ++it) {
    delete it->yuvpbo;
    it->reset();
  }
  for(auto it=infifo.reservoir_1440p.begin(); it!=infifo.reservoir_1440p.end(); ++it) {
    delete it->yuvpbo;
    it->reset();
  }
  for(auto it=infifo.reservoir_4K.begin(); it!=infifo.reservoir_4K.end(); ++it) {
    delete it->yuvpbo;
    it->reset();
  }
  glFinish();
}


void OpenGLThread::initGLX() {  
  // GLXFBConfig *fbConfigs;
  int numReturned;
  
  // initial connection to the xserver
  this->display_id = XOpenDisplay(NULL);
  if (this->display_id == NULL) {
    opengllogger.log(LogLevel::fatal) << "OpenGLThtead: initGLX: cannot connect to X server" << std::endl;
  }
  
  // glx frame buffer configuration [GLXFBConfig * list of GLX frame buffer configuration parameters] => consistent visual [XVisualInfo] parameters for the X-window
  this->root_id =DefaultRootWindow(this->display_id); // get the root window of this display
    
  /* Request a suitable framebuffer configuration - try for a double buffered configuration first */
  this->doublebuffer_flag=true;
  this->fbConfigs = glXChooseFBConfig(this->display_id,DefaultScreen(this->display_id),glx_attr::doubleBufferAttributes,&numReturned);
  // MEMORY LEAK when running with valgrind, see: http://stackoverflow.com/questions/10065849/memory-leak-using-glxcreatecontext
  
  this->att=glx_attr::doubleBufferAttributes;
  // this->fbConfigs = NULL; // force single buffer
  
  if (this->fbConfigs == NULL) {  /* no double buffered configs available */
    this->fbConfigs = glXChooseFBConfig( this->display_id, DefaultScreen(this->display_id),glx_attr::singleBufferAttributes,&numReturned);
    this->doublebuffer_flag=False;
    this->att=glx_attr::singleBufferAttributes;
  }
    
  if (this->fbConfigs == NULL) {
    opengllogger.log(LogLevel::fatal) << "OpenGLThread: initGLX: WARNING! no GLX framebuffer configuration" << std::endl;
  }

  this->glc=glXCreateNewContext(this->display_id,this->fbConfigs[0],GLX_RGBA_TYPE,NULL,True);
  if (!this->glc) {
    opengllogger.log(LogLevel::fatal) << "OpenGLThread: initGLX: FATAL! Could not create glx context"<<std::endl; 
    exit(2);
  }
  // this->glc=glXCreateNewContext(this->display_id,fbConfigs[0],GLX_RGBA_TYPE,NULL,False); // indirect rendering!
  // printf("ContextManager: glx context handle   =%lu\n",this->glc);
  // XFree(this->fbConfigs);
  
  // glXSwapIntervalEXT(0); // we dont have this..
  // PFNGLXSWAPINTERVALEXTPROC glXSwapIntervalEXT = (PFNGLXSWAPINTERVALEXTPROC)glXGetProcAddress((const GLubyte*)"glXSwapIntervalEXT");  // Set the glxSwapInterval to 0, ie. disable vsync!  khronos.org/opengl/wiki/Swap_Interval
  // glXSwapIntervalEXT(display_id, root_id, 0);  // glXSwapIntervalEXT(0); // not here ..
  
}


void OpenGLThread::makeCurrent(Window window_id) {
  glXMakeCurrent(this->display_id, window_id, this->glc);
}


unsigned OpenGLThread::getVsyncAtStartup() {
  return swap_interval_at_startup;
}


int OpenGLThread::hasCompositor(int screen) {
  // https://stackoverflow.com/questions/33195570/detect-if-compositor-is-running
  char prop_name[20];
  snprintf(prop_name, 20, "_NET_WM_CM_S%d", screen);
  Atom prop_atom = XInternAtom(display_id, prop_name, False);
  return XGetSelectionOwner(display_id, prop_atom) != None;
}


void OpenGLThread::closeGLX() {
  XFree(this->fbConfigs);
  glXDestroyContext(this->display_id, this->glc);
  XCloseDisplay(this->display_id);
}


void OpenGLThread::loadExtensions() {
  if (GLEW_ARB_pixel_buffer_object) {
    opengllogger.log(LogLevel::debug) << "OpenGLThread: loadExtensions: PBO extension already loaded" <<std::endl;
    return;
  }
  else {
    opengllogger.log(LogLevel::crazy) << "OpenGLThread: loadExtensions: Will load PBO extension" <<std::endl;
  }
  
  this->makeCurrent(this->root_id); // a context must be made current before glew works..
  
  glewExperimental = GL_TRUE;
  GLenum err = glewInit();
  if (GLEW_OK != err) {
  /* Problem: glewInit failed, something is seriously wrong. */
  opengllogger.log(LogLevel::fatal) << "OpenGLThread: loadExtensions: ERROR: " << glewGetErrorString(err) <<std::endl;  
  }
  else {
    if (GLEW_ARB_pixel_buffer_object) {
    opengllogger.log(LogLevel::debug) << "OpenGLThread: loadExtensions:  PBO extension found! :)"<<std::endl;
    }
    else {
      opengllogger.log(LogLevel::fatal) << "OpenGLThread: loadExtensions: WARNING: PBO extension not found! :("<<std::endl;
    }
  }
  
  // load glx extensions
  ///*
  if (is_glx_extension_supported(display_id, "GLX_EXT_swap_control")) {
      std::cout << "GLX_EXT_swap_control" << std::endl;
      // typedef void ( *PFNGLXSWAPINTERVALEXTPROC) (Display *dpy, GLXDrawable drawable, int interval);
      this->pglXSwapIntervalEXT =
        (PFNGLXSWAPINTERVALEXTPROC) 
        glXGetProcAddressARB((const GLubyte *) "glXSwapIntervalEXT");
      // perror("OpenGLThread: loadExtensions: GLX_EXT_swap_control: GXL_MESA_swap_control required!");
      swap_flavor =swap_flavors::ext;
  }
  //*/
  else if (is_glx_extension_supported(display_id, "GLX_MESA_swap_control")) {
    // std::cout << "GLX_MESA_swap_control" << std::endl;
    // typedef int (*PFNGLXGETSWAPINTERVALMESAPROC)(void);
    // typedef int (*PFNGLXSWAPINTERVALMESAPROC)(unsigned int interval);
    
    // PFNGLXGETSWAPINTERVALMESAPROC pglXGetSwapIntervalMESA =
    this->pglXGetSwapIntervalMESA =
        (PFNGLXGETSWAPINTERVALMESAPROC)
        glXGetProcAddressARB((const GLubyte *) "glXGetSwapIntervalMESA");
        
    // PFNGLXSWAPINTERVALMESAPROC pglXSwapIntervalMESA =
    this->pglXSwapIntervalMESA =
        (PFNGLXSWAPINTERVALMESAPROC)
        glXGetProcAddressARB((const GLubyte *) "glXSwapIntervalMESA"); // actually.. this seems to be in glx.h and not in glxext.h anymore..
        
    //interval = (*pglXGetSwapIntervalMESA)();
    swap_flavor =swap_flavors::mesa;
  } 
  else if (is_glx_extension_supported(display_id, "GLX_SGI_swap_control")) {
    std::cout << "GLX_SGI_swap_control" << std::endl;
    /* The default swap interval with this extension is 1.  Assume that it
      * is set to the default.
      *
      * Many Mesa-based drivers default to 0, but all of these drivers also
      * export GLX_MESA_swap_control.  In that case, this branch will never
      * be taken, and the correct result should be reported.
      */
    // interval = 1;
    
    // eh.. this is basically useless
    perror("OpenGLThread: loadExtensions: GLX_SGI_swap_control: there's something wrong with your graphics driver");
    // swap_flavor =swap_flavors::sgi;
    swap_flavor =swap_flavors::none;
   }
  else {
    perror("OpenGLThread: loadExtensions: no swap control: there's something wrong with your graphics driver");
    swap_flavor =swap_flavors::none;
  }
}
  
  
void OpenGLThread::VSyncOff() {
  setSwapInterval(0);
}
  

void OpenGLThread::makeShaders() {
  
  opengllogger.log(LogLevel::debug) << "OpenGLThread: makeShaders: compiling YUVShader" << std::endl;
  yuv_shader =new YUVShader();
  /*
  opengllogger.log(LogLevel::debug) << "OpenGLThread: makeShaders: compiling RGBShader" << std::endl;
  rgb_shader =new RGBShader();
  */
  if (opengllogger.log_level>=LogLevel::debug) {
    yuv_shader->validate();
    // rgb_shader->validate();
  }
}


void OpenGLThread::delShaders() {
  delete yuv_shader;
  // delete rgb_shader;
}
  

Window OpenGLThread::createWindow(bool map) {
  Window win_id;
  XSetWindowAttributes swa;
  
  // this->vi  =glXChooseVisual(this->display_id, 0, this->att); // "visual parameters" of the X window
  this->vi =glXGetVisualFromFBConfig( this->display_id, this->fbConfigs[0] ); // another way to do it ..
  
  swa.colormap   =XCreateColormap(this->display_id, this->root_id, (this->vi)->visual, AllocNone);
  // swa.event_mask =ExposureMask | KeyPressMask;
  swa.event_mask =NoEventMask;
  // swa.event_mask =ButtonPressMask|ButtonReleaseMask;
  // swa.event_mask =ExposureMask|KeyPressMask|KeyReleaseMask|ButtonPressMask|ButtonReleaseMask|StructureNotifyMask;
  
  win_id =XCreateWindow(this->display_id, this->root_id, 0, 0, 600, 600, 0, vi->depth, InputOutput, vi->visual, CWColormap | CWEventMask, &swa);
  
  XStoreName(this->display_id, win_id, "test window");
  if (map) {
    XMapWindow(this->display_id, win_id);
  }
  
  // win_id =glXCreateWindow(this->display_id, this->fbConfigs[0], win_id, NULL);

  // makeCurrent(win_id);
  return win_id;
}


void OpenGLThread::reConfigWindow(Window window_id) {
  // glXSwapIntervalEXT(display_id, window_id, 0); // segfaults ..
  XSetWindowAttributes swa;
  // swa.colormap   =XCreateColormap(this->display_id, this->root_id, (this->vi)->visual, AllocNone);
  swa.event_mask =NoEventMask;
  // swa.event_mask =ExposureMask|KeyPressMask|KeyReleaseMask|ButtonPressMask|ButtonReleaseMask|StructureNotifyMask;
  // XChangeWindowAttributes(display_id, window_id, valuemask, attributes)
  // XChangeWindowAttributes(this->display_id, window_id, CWColormap | CWEventMask, &swa); // crashhhh
  XChangeWindowAttributes(this->display_id, window_id, CWEventMask, &swa); // ok ..
  // return glXCreateWindow(this->display_id, this->fbConfigs[0], window_id, NULL); // what crap is this..?
  // return window_id;
}


Window OpenGLThread::getChildWindow(Window parent_id) { // create new x window as a child of window_id
  // https://forum.qt.io/topic/34165/x11-reparenting-does-not-work
  Window child_id;
  
  child_id=createWindow();
  XReparentWindow(this->display_id, child_id, parent_id, 0, 0);
  // XReparentWindow(this->display_id, parent_id, child_id, 0, 0); // nopes ..
  // XMapWindow(this->display_id, child_id); 
  XFlush(this->display_id);
  return child_id;
}


void OpenGLThread::reportSlots() {
  std::cout << "OpenGLThread: reportSlots:" << std::endl;
  int i;
  
  /*
  for(i=0;i<int(N_MAX_SLOTS);i++) {
    std::cout << slots_[i].isActive() << std::endl;
    // std::cout << ">>"<<slots_.back().isActive()<<std::endl;
  }
  return;
  */
  
  i=0;
  std::cout << "OpenGLThread: reportSlots: Activated slots:" << std::endl;
  for(std::vector<SlotContext>::iterator it=slots_.begin(); it!=slots_.end(); ++it) {
    if (it->isActive()) {
      std::cout << "OpenGLThread: reportSlots: ACTIVE "<<i<<std::endl;
    }
    i++;
  }
  std::cout << "OpenGLThread: reportSlots:" << std::endl;
}


void OpenGLThread::reportRenderGroups() {
  std::cout << "OpenGLThread: reportRenderGroups: " << std::endl;
  std::cout << "OpenGLThread: reportRenderGroups: Render groups and their render contexes:" << std::endl;
  for(std::map<Window,RenderGroup>::iterator it=render_groups.begin(); it!=render_groups.end(); ++it) {
    std::cout << "OpenGLThread: reportRenderGroups: x window id "<< it->first << std::endl;
  }
  std::cout << "OpenGLThread: reportRenderGroups: " << std::endl;
}
  
  
void OpenGLThread::reportRenderList() {
  int i=0;
  std::cout << "OpenGLThread: reportRenderList: " << std::endl;
  std::cout << "OpenGLThread: reportRenderList: grouped render contexes" << std::endl;
  
  for(std::vector<std::list<RenderGroup*>>::iterator it=render_lists.begin(); it!=render_lists.end(); ++it) {
    if (it->size()>0) { // slot has render groups
      std::cout << "OpenGLThread: reportRenderList: groups at slot " << i <<std::endl;
      
      for(std::list<RenderGroup*>::iterator it2=it->begin(); it2!=it->end(); ++it2) {
        // std::cout << "OpenGLThread: reportRenderList:   x window id "<<(*it2)->getWindowId()<<" with "<<(*it2)->render_contexes.size() << " contexes "<<std::endl;
        std::cout << "OpenGLThread: reportRenderList:   x window id "<<(*it2)->getWindowId()<<std::endl;
        
        std::list<RenderContext> render_contexes=(*it2)->getRenderContexes();
        
        for (std::list<RenderContext>::iterator it3=render_contexes.begin(); it3!=render_contexes.end(); ++it3) {
          std::cout << "OpenGLThread: reportRenderList:     * render context "<< it3->getId() << std::endl;
        } // render contexes
        
      } // render groups
      
    } // if slot has render groups
    i++;
  } // slots
  std::cout << "OpenGLThread: reportRenderList: " << std::endl;
  // std::cout << "OpenGLThread: reportRenderList: " << render_lists[1].size() << std::endl;
}


// API

void OpenGLThread::stopCall() {
  SignalContext signal_ctx;
  
  signal_ctx.signal=Signals::exit;
  sendSignal(signal_ctx);
  this->closeThread();
  this->has_thread=false;
}


void OpenGLThread::infoCall() {
  SignalContext signal_ctx;
  
  signal_ctx.signal=Signals::info;
  sendSignal(signal_ctx);
}


bool OpenGLThread::newRenderGroupCall  (Window window_id) { // return value: render_ctx
  OpenGLSignalContext ctx;
  
  ctx.n_slot      =0;
  ctx.x_window_id =window_id;
  ctx.z           =0;
  ctx.render_ctx  =0;
  ctx.success     =false;
  
  opengllogger.log(LogLevel::debug) << "OpenGLThread: newRenderCroupCall: ctx="<< ctx <<std::endl;
  
  SignalContext signal_ctx = {Signals::new_render_group, &ctx};
  // eh.. we're passing pointer ctx .. but ctx goes null once we get out of context..!
  // sendSignal(signal_ctx); // we must keep ctx alive, so use:
  sendSignalAndWait(signal_ctx);
  
  opengllogger.log(LogLevel::debug) << "OpenGLThread: newRenderCroupCall: return ctx="<< ctx <<std::endl;
  return ctx.success;
}


bool OpenGLThread::delRenderGroupCall  (Window window_id) {
  OpenGLSignalContext ctx;
  
  ctx.n_slot      =0;
  ctx.x_window_id =window_id;
  ctx.z           =0;
  ctx.render_ctx  =0;
  ctx.success     =false;
  
  SignalContext signal_ctx = {Signals::del_render_group, &ctx};
  sendSignalAndWait(signal_ctx);
  
  opengllogger.log(LogLevel::debug) << "OpenGLThread: delRenderCroupCall: return ctx="<< ctx <<std::endl;
  return ctx.success;
}
  

int OpenGLThread::newRenderContextCall(SlotNumber slot, Window window_id, unsigned int z) {
  OpenGLSignalContext ctx;
  
  ctx.n_slot      =slot;
  ctx.x_window_id =window_id;
  ctx.z           =z;
  ctx.render_ctx  =0; // return value
  ctx.success     =false;
  
  SignalContext signal_ctx = {Signals::new_render_context, &ctx};
  sendSignalAndWait(signal_ctx);
  // there could be a mutex going in with the signal .. and then we wait for that mutex
  
  opengllogger.log(LogLevel::debug) << "OpenGLThread: newRenderContextCall: return ctx="<< ctx <<std::endl;
  
  /* // TODO: check this!
  if (!ctx.success) {
    return 0;
  }
  */
  
  return ctx.render_ctx;
}
  
  
bool OpenGLThread::delRenderContextCall(int id) {
  OpenGLSignalContext ctx;
  
  ctx.n_slot      =0;
  ctx.x_window_id =0;
  ctx.z           =0;
  ctx.render_ctx  =id; // input
  ctx.success     =false;
  
  SignalContext signal_ctx = {Signals::del_render_context, &ctx};
  sendSignalAndWait(signal_ctx);

  opengllogger.log(LogLevel::debug) << "OpenGLThread: delRenderContextCall: return ctx="<< ctx <<std::endl;
  return ctx.success;
}


