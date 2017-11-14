/*
 * opengl.cpp : X11, GLX, OpenGL calls for initialization and texture dumping, plus some auxiliary routines
 * 
 * Copyright 2017 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Valkka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Valkka.  If not, see <http://www.gnu.org/licenses/>. 
 * 
 */

/** 
 *  @file    opengl.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
 *  
 *  @brief X11, GLX, OpenGL calls for initialization and texture dumping, plus some auxiliary routines
 *
 *  @section DESCRIPTION
 *  
 */ 
 
#include "opengl.h"
#include "logging.h"

// WARNING: these define switches should be off (commented) by default
// #define VALGRIND_GPU_DEBUG 1 // enable this for valgrind debugging.  Otherwise direct memory access to GPU drives it crazy.
// #define LOAD_VERBOSE 1 // shows information on loading YUVPBO and TEX structures

// Auxiliary routine for reading a bitmap file
int readbytes(const char* fname, char*& buffer) {
  using std::ios;
  using std::ifstream;
  
  int      size;
  ifstream file;
  
  file.open(fname,ios::in|ios::binary|ios::ate);
  size = file.tellg();
  buffer = new char[size];
  file.seekg(0,ios::beg);
  file.read(buffer,size);
  file.close();
  std::cout << "read "<<size<<" bytes"<<std::endl;
  
  return size;
}


// Auxiliary routine for reading a YUV bitmap file
int readyuvbytes(const char* fname, GLubyte*& Y, GLubyte*& U, GLubyte*& V) {
  using std::ios;
  using std::ifstream;
  GLubyte*  buffer;
  // char* buffer;
  int      size, ysize;
  ifstream file;
  
  file.open(fname,ios::in|ios::binary|ios::ate);
  size = file.tellg();
  buffer = new GLubyte[size];
  // buffer = new char[size];
  file.seekg(0,ios::beg);
  file.read((char*)buffer,size);
  // file.read(buffer,size);
  file.close();
  
  ysize=(size*2)/3; // size of the biggest plane (Y)
  
  /*
  std::cout << "size="<<size<<" ysize="<<ysize<<std::endl;
  int i;
  for(i=0;i<size;i++) {
    std::cout << (unsigned) buffer[i] << " ";
  }
  return 0;
  */
  
  Y=buffer;
  U=buffer+ysize;
  V=buffer+ysize+ysize/4;
  
  /*
  for(i=0;i<std::min(ysize,20);i++) {
    std::cout << U[i] << " ";
  }
  std::cout << std::endl;
  */
  
  std::cout << "read "<<size<<" bytes"<<std::endl;
  
  return size;
}

void zeroyuvbytes(int ysize, GLubyte*& Y, GLubyte*& U, GLubyte*& V) { // debugging of debugging ..
  int i;
  
  for(i=0;i<ysize;i++) {
    Y[i]=0;
  }
  
  for(i=0;i<ysize/4;i++) {
    U[i]=0;
    V[i]=0;
  }
  
}


#ifdef VALGRIND_GPU_DEBUG
// This is for debugging.. it seems that valgrind gives false positives when copying into gpu memory
// https://www.opengl.org/discussion_boards/showthread.php/175407-glBindFramebuffer-memory-leak
 
void getPBO(GLuint& index, GLsizei size, GLubyte*& payload) {
  index=1;
  payload = new GLubyte[size];
  std::cout << "fake getPBO: size="<<size<<std::endl;
}

void releasePBO(GLuint* index, GLubyte* payload) {
  delete[] payload;
}

#else

void getPBO(GLuint& index, GLsizei size, GLubyte*& payload) { // modify pointer in-place
  // WARNING! load openGL extensions before using this! (i.e., use OpenGLThread.preRun)
  glGenBuffersARB(1, &index);
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, index);
  glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, size, 0, GL_STREAM_DRAW_ARB); // reserve n_payload bytes to index/handle pbo_id
  
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0); // unbind (not mandatory)
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, index); // rebind (not mandatory)
  
  payload = (GLubyte*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
  
  glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); // release pointer to mapping buffer ** MANDATORY **
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0); // unbind ** MANDATORY **
}

void releasePBO(GLuint* index, GLubyte* payload) {
  glDeleteBuffersARB(1, index);
}

#endif


void getTEX(GLuint& index, GLint format, GLsizei w, GLsizei h) {
  // for YUV: texture_format=GL_RED;
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &index);
  
  glBindTexture(GL_TEXTURE_2D, index);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, 0); // no upload, just reserve 
  glBindTexture(GL_TEXTURE_2D, 0); // unbind
}


void loadYUVPBO(YUVPBO* pbo, GLsizei size, GLubyte* y, GLubyte* u, GLubyte* v) {
  unsigned int i;
  
  i=std::min(size,pbo->size);
  
#ifdef LOAD_VERBOSE
  std::cout << "loadYUVPBO: pbo= "<< *pbo << std::endl;
  std::cout << "loadYUVPBO: i  ="<< i << std::endl;
#endif
  
  // memcpy(pbo->y_payload, y, 1); // debugging
  memcpy(pbo->y_payload, y, i);
  // /*
  memcpy(pbo->u_payload, u, i/4);
  memcpy(pbo->v_payload, v, i/4);
  // */
}

void peekYUVPBO(YUVPBO* pbo) {
  int i;
  
  std::cout << "peekYUVPBO: y_payload: ";
  for(i=0; i<std::min(20,pbo->size); i++) {
    std::cout << (unsigned int) pbo->y_payload[i] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "peekYUVPBO: u_payload: ";
  for(i=0; i<std::min(20,pbo->size/4); i++) {
    std::cout << (unsigned int) pbo->u_payload[i] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "peekYUVPBO: v_payload: ";
  for(i=0; i<std::min(20,pbo->size/4); i++) {
    std::cout << (unsigned int) pbo->v_payload[i] << " ";
  }
  std::cout << std::endl;
}


/** Copy pixel buffer object to texture
 * 
 */
void loadYUVTEX(YUVPBO* pbo, YUVTEX* tex) {

#ifdef LOAD_VERBOSE
  std::cout << std::endl << "loadYUVTEX: pbo      : " << *pbo << std::endl;
  std::cout << "loadYUVTEX: pbo peek : " << std::endl;
  peekYUVPBO(pbo);
  std::cout << "loadYUVTEX: tex      : " << *tex << std::endl;
#endif
  
  // y
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo->y_index);
  glBindTexture(GL_TEXTURE_2D, tex->y_index); // this is the texture we will manipulate
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex->w, tex->h, tex->format, GL_UNSIGNED_BYTE, 0); // copy from pbo to texture 
  // glBindTexture(GL_TEXTURE_2D, 0); 
  // u
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo->u_index);
  glBindTexture(GL_TEXTURE_2D, tex->u_index); // this is the texture we will manipulate
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex->w/2, tex->h/2, tex->format, GL_UNSIGNED_BYTE, 0); // copy from pbo to texture 
  // glBindTexture(GL_TEXTURE_2D, 0);  
  // v
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo->v_index);
  glBindTexture(GL_TEXTURE_2D, tex->v_index); // this is the texture we will manipulate
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex->w/2, tex->h/2, tex->format, GL_UNSIGNED_BYTE, 0); // copy from pbo to texture 
  // glBindTexture(GL_TEXTURE_2D, 0); 
  
  // TODO: implement also pipeline going the other way, i.e.: render to texture => texture to PBO => download data from PBO using dma
  //
  // http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-14-render-to-texture/
  // https://www.khronos.org/opengl/wiki/Pixel_Buffer_Object
  
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0); // unbind // important!
  glBindTexture(GL_TEXTURE_2D, 0); // unbind
  
  
}


// YUVPBO::YUVPBO(GLsizei size) : size(size) {
YUVPBO::YUVPBO(BitmapType bmtype) : bmtype(bmtype) {
  switch(bmtype) {
    case (BitmapPars::N720::type): {
      size=BitmapPars::N720::size;
      break;
    }
    case (BitmapPars::N1080::type): {
      size=BitmapPars::N1080::size;
      break;
    }
    case (BitmapPars::N1440::type): {
      size=BitmapPars::N1440::size;
      break;
    }
    case (BitmapPars::N4K::type): {
      size=BitmapPars::N4K::size;
      break;
    }
    default: {
      opengllogger.log(LogLevel::fatal) << "YUVPBO: FATAL! No such bitmap type "<<bmtype<<std::endl;
      size=0;
      break;
    }
  }
  
  // std::cout<<">>>> size="<<size<<std::endl;
  
  reserve();
}


YUVPBO::~YUVPBO() {
  opengllogger.log(LogLevel::debug) << "YUVPBO: destructor"<<std::endl;
  releasePBO(&y_index, y_payload);
  releasePBO(&u_index, u_payload);
  releasePBO(&v_index, v_payload);
  opengllogger.log(LogLevel::debug) << "YUVPBO: destructor: bye"<<std::endl;
}


void YUVPBO::reserve() {
  bool ok=true;
  
  getPBO(y_index,size,y_payload);
  if (y_payload) {
    opengllogger.log(LogLevel::debug) << "YUVPBO: reserve: Y: Got databuf pbo_id " <<y_index<< " with adr="<<(long unsigned)(y_payload)<<std::endl;
  } else {ok=false;}
  
  getPBO(u_index,size/4,u_payload);
  if (u_payload) {
    opengllogger.log(LogLevel::debug) << "YUVPBO: reserve: U: Got databuf pbo_id " <<u_index<< " with adr="<<(long unsigned)(u_payload)<<std::endl;
  } else {ok=false;}
  
  getPBO(v_index,size/4,v_payload);
  if (v_payload) {
    opengllogger.log(LogLevel::debug) << "YUVPBO: reserve: V: Got databuf pbo_id " <<v_index<< " with adr="<<(long unsigned)(v_payload)<<std::endl;
  } else {ok=false;}
  
  if (!ok) {
    opengllogger.log(LogLevel::fatal) << "YUVPBO: reserve: WARNING: could not get GPU ram (out of memory?) "<<std::endl;
    perror("YUVPBO: reserve: WARNING: could not get GPU ram (out of memory?)");
  }
  
}


void YUVPBO::upload(GLsizei isize, GLubyte* y, GLubyte* u, GLubyte* v) {
  // loadYUVPBO(YUVPBO* pbo, GLsizei size, GLubyte* y, GLubyte* u, GLubyte* v); // let's rewrite it here..
  GLsizei i;
  i=std::min(isize,size);
  // memcpy(pbo->y_payload, y, 1); // debugging
  memcpy(y_payload, y, i);
  // /*
  memcpy(u_payload, u, i/4);
  memcpy(v_payload, v, i/4);
  // */
}



TEX::TEX(GLsizei w, GLsizei h) : format(0), w(w), h(h), index(0) {
}
  

TEX::~TEX() {
}
  

  
YUVTEX::YUVTEX(GLsizei w, GLsizei h) : TEX(w, h), y_index(0), u_index(0), v_index(0) {
  opengllogger.log(LogLevel::crazy) << "YUVTEX: reserving" << std::endl;
  this->format=GL_RED;
  getTEX(this->y_index, this->format, this->w,   this->h);
  getTEX(this->u_index, this->format, this->w/2, this->h/2);
  getTEX(this->v_index, this->format, this->w/2, this->h/2);
  opengllogger.log(LogLevel::crazy) << "YUVTEX: reserved " << *this;
  glFinish();
}

YUVTEX::~YUVTEX() {
  opengllogger.log(LogLevel::crazy) << "YUVTEX: releasing " << *this;
  glDeleteTextures(1, &(this->y_index));
  glDeleteTextures(1, &(this->u_index));
  glDeleteTextures(1, &(this->v_index));
  glFinish();
}
  
  
  
std::ostream &operator<<(std::ostream &os, YUVPBO const &m) {
 return os << "<y size=" <<m.size << " y ["<<m.y_index<<" "<<(unsigned long)m.y_payload<<"] " << "u ["<<m.u_index<<" "<<(unsigned long)m.u_payload<<"] " << "v ["<<m.v_index<<" "<<(unsigned long)m.v_payload<<"]> ";
}


std::ostream &operator<<(std::ostream &os, YUVTEX const &m) {
  return os << "<w=" <<m.w<<" h="<<m.h<<" tex refs=["<<m.y_index<<" "<<m.u_index<<" "<<m.v_index<<"]>" << std::endl;
}




