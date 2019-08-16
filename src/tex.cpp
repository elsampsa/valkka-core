/*
 * opengl.cpp : X11, GLX, OpenGL calls for initialization and texture dumping, plus some auxiliary routines
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
 *  @file    opengl.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.13.2 
 *  
 *  @brief X11, GLX, OpenGL calls for initialization and texture dumping, plus some auxiliary routines
 *
 *  @section DESCRIPTION
 *  
 */ 
 
#include "tex.h"
#include "logging.h"
#include "tools.h"


// TEX::TEX(BitmapPars bmpars) :  bmpars(bmpars), source_bmpars(bmpars), index(0), format(0) {
TEX::TEX(BitmapPars bmpars) :  bmpars(bmpars), index(0), format(0) {
}
  
TEX::~TEX() {
}


  
YUVTEX::YUVTEX(BitmapPars bmpars) : TEX(bmpars), y_index(0), u_index(0), v_index(0) {
  opengllogger.log(LogLevel::crazy) << "YUVTEX: reserving" << std::endl;
  // this->format=GL_RED; // TODO: is this deprecated or what .. ?
  // this->format=GL_COMPRESSED_RED_RGTC1; // fooling around
  // this->format=GL_R8; // fooling around
  // this->format=GL_RGB; // just fooling around ..
  
  // this->format             =GL_DEPTH_COMPONENT;
  // this->internal_format    =GL_R8;
  
  format             =GL_RED;
  internal_format    =GL_RED;
  
#ifdef VALGRIND_GPU_DEBUG
#else
  getTEX(y_index, internal_format, format, bmpars.y_width, bmpars.y_height);
  getTEX(u_index, internal_format, format, bmpars.u_width, bmpars.u_height);
  getTEX(v_index, internal_format, format, bmpars.v_width, bmpars.v_height);
  glFinish();
#endif
  opengllogger.log(LogLevel::crazy) << "YUVTEX: reserved " << *this;
}

YUVTEX::~YUVTEX() {
  opengllogger.log(LogLevel::crazy) << "YUVTEX: releasing " << *this;
#ifdef VALGRIND_GPU_DEBUG
#else
  glDeleteTextures(1, &(y_index));
  glDeleteTextures(1, &(u_index));
  glDeleteTextures(1, &(v_index));
  glFinish();
#endif
}
  
  
/** Load texture from memory buffers
 * 
 * 
 */
void YUVTEX::loadYUV(const GLubyte* Y, const GLubyte* U, const GLubyte* V) {
  // y
  glBindTexture(GL_TEXTURE_2D, y_index); // this is the texture we will manipulate
  glTexImage2D(GL_TEXTURE_2D, 0, format, bmpars.y_width, bmpars.y_height, 0, format, GL_UNSIGNED_BYTE, Y);
  // glBindTexture(GL_TEXTURE_2D, 0); 
  
  /*
  std::cout << "YUVTEX: loadYUV: y_index =" << y_index << std::endl;
  int i;
  for(i=0;i<100;i++) {
    std::cout << (unsigned) Y[i] << " ";
  }
  */
  
  // u
  glBindTexture(GL_TEXTURE_2D, u_index); // this is the texture we will manipulate
  glTexImage2D(GL_TEXTURE_2D, 0, format, bmpars.u_width, bmpars.u_height, 0, format, GL_UNSIGNED_BYTE, U);
  // glBindTexture(GL_TEXTURE_2D, 0); 
  
  // v
  glBindTexture(GL_TEXTURE_2D, v_index); // this is the texture we will manipulate
  glTexImage2D(GL_TEXTURE_2D, 0, format, bmpars.v_width, bmpars.v_height, 0, format, GL_UNSIGNED_BYTE, V);
  // glBindTexture(GL_TEXTURE_2D, 0); 
  
  glFinish();
}
  

/** Copy pixel buffer object to texture
 * 
 * Texture and YUVFrame are assumed to have the same dimensions
 * 
 */
void YUVTEX::loadYUVFrame(YUVFrame *yuvframe) {
  if (bmpars.type!=yuvframe->source_bmpars.type) {
    opengllogger.log(LogLevel::fatal) << "YUVTEX: loadYUVFrame: FATAL: inconsistent YUVFrame and TEX types!" << std::endl;
    exit(5);
  }
  
  // bmpars are the bitmap parameters for this texture
  // they correspond to YUVFrame->source_bmpars
  
#ifdef LOAD_VERBOSE
  std::cout << std::endl << "YUVTEX: loadYUVFrame: frame: " << *yuvframe << std::endl;
  std::cout << yuvframe->dumpPayload();
  std::cout << "YUVTEX: loadYUVFrame:  tex: " << *this << std::endl;
#endif
  
#ifdef OPENGL_TIMING
  long int mstime =getCurrentMsTimestamp();
  long int swaptime;
#endif
   
#ifdef VALGRIND_GPU_DEBUG
#else
  // y
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, yuvframe->y_index);
  glBindTexture(GL_TEXTURE_2D, y_index); // this is the texture we will manipulate
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, bmpars.y_width, bmpars.y_height, format, GL_UNSIGNED_BYTE, 0); // copy from pbo to texture 
  glBindTexture(GL_TEXTURE_2D, 0); 
  // glFlush();
  // glFinish();
  
  // u
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, yuvframe->u_index);
  glBindTexture(GL_TEXTURE_2D, u_index); // this is the texture we will manipulate
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, bmpars.u_width, bmpars.u_height, format, GL_UNSIGNED_BYTE, 0); // copy from pbo to texture 
  glBindTexture(GL_TEXTURE_2D, 0);  
  // glFlush();
  // glFinish();
  
  // v
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, yuvframe->v_index);
  glBindTexture(GL_TEXTURE_2D, v_index); // this is the texture we will manipulate
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, bmpars.v_width, bmpars.v_height, format, GL_UNSIGNED_BYTE, 0); // copy from pbo to texture 
  glBindTexture(GL_TEXTURE_2D, 0); 
  // glFlush();
  // glFinish();
  
  // TODO: implement also pipeline going the other way, i.e.: render to texture => texture to PBO => download data from PBO using dma
  //
  // http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-14-render-to-texture/
  // https://www.khronos.org/opengl/wiki/Pixel_Buffer_Object
  
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // unbind // important!
  glBindTexture(GL_TEXTURE_2D, 0); // unbind
  
  // glFlush();
  glFinish(); // TODO: debugging
#endif
  
#ifdef OPENGL_TIMING
  swaptime=mstime; mstime=getCurrentMsTimestamp();
  if ( (mstime-swaptime) > 2 ) {
    std::cout << "YUVTEX: loadYUVFrame: timing : " << mstime-swaptime << std::endl;
  }
#endif

  // source_bmpars=yuvframe->source_bmpars; // copy the actual bitmap dimensions
}
  
  
std::ostream &operator<<(std::ostream &os, YUVTEX const &m) {
  return os << "<w=" <<m.bmpars.width<<" h="<<m.bmpars.height<<" tex refs=["<<m.y_index<<" "<<m.u_index<<" "<<m.v_index<<"]>" << std::endl;
}



