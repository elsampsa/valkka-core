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
 *  @version 0.5.1 
 *  
 *  @brief X11, GLX, OpenGL calls for initialization and texture dumping, plus some auxiliary routines
 *
 *  @section DESCRIPTION
 *  
 */ 
 
#include "opengl.h"
#include "logging.h"
#include "tools.h"


/** // ripped off from glxgears.c
 * Determine whether or not a GLX extension is supported.
 */
int is_glx_extension_supported(Display *dpy, const char *query)
{
   const int scrnum = DefaultScreen(dpy);
   const char *glx_extensions = NULL;
   const size_t len = strlen(query);
   const char *ptr;

   if (glx_extensions == NULL) {
    glx_extensions = glXQueryExtensionsString(dpy, scrnum);
   }

   ptr = strstr(glx_extensions, query);
   return ((ptr != NULL) && ((ptr[len] == ' ') || (ptr[len] == '\0')));
}


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
  glGenBuffers(1, &index);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, index);
  // glBufferData(GL_PIXEL_UNPACK_BUFFER, size, 0, GL_STREAM_DRAW); // reserve n_payload bytes to index/handle pbo_id
  glBufferData(GL_PIXEL_UNPACK_BUFFER, size, 0, GL_DYNAMIC_DRAW); // reserve n_payload bytes to index/handle pbo_id // this should be better ..
  
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // unbind (not mandatory)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, index); // rebind (not mandatory)
  
  payload = (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
  
  glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER); // release pointer to mapping buffer ** MANDATORY **
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // unbind ** MANDATORY **
  
  glFinish();
  glFlush();
  memset(payload, 0, size); // warm up the gpu memory..
}

void releasePBO(GLuint* index, GLubyte* payload) {
  opengllogger.log(LogLevel::crazy) << "releasePBO: released " << (unsigned long)payload << std::endl;
  glDeleteBuffers(1, index);
}

#endif


void getTEX(GLuint& index, GLint internal_format, GLint format, GLsizei w, GLsizei h) {
  // for YUV: texture_format=GL_RED;
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &index);
  
  glBindTexture(GL_TEXTURE_2D, index);
  
  //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  
  // faster?  makes more sense anyway
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // turbo
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // for large images, interpolate
  
  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, format, GL_UNSIGNED_BYTE, 0); // no upload, just reserve 
  glBindTexture(GL_TEXTURE_2D, 0); // unbind
}



