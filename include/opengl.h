#ifndef OPENGL_HEADER_GUARD 
#define OPENGL_HEADER_GUARD

/*
 * opengl.h : X11, GLX, OpenGL calls for initialization and texture dumping, plus some auxiliary routines
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
 *  @file    opengl.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.2.0 
 *  
 *  @brief X11, GLX, OpenGL calls for initialization and texture dumping, plus some auxiliary routines
 *  
 */ 

#include "common.h"
#include "sizes.h"
#include<GL/glew.h>
#include<GL/glx.h>
#include<GL/glxext.h>

int is_glx_extension_supported(Display *dpy, const char *query);

/** GLX parameter groups
 * @ingroup openglthread_tag
 */
namespace glx_attr { // https://stackoverflow.com/questions/11623451/static-vs-non-static-variables-in-namespace
  static int singleBufferAttributes[] = {
    GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
    GLX_RENDER_TYPE,   GLX_RGBA_BIT,
    GLX_RED_SIZE,      1,   // Request a single buffered color buffer
    GLX_GREEN_SIZE,    1,   // with the maximum number of color bits
    GLX_BLUE_SIZE,     1,   // for each component                     
    None
  };
  static int doubleBufferAttributes[] = {
    GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
    GLX_RENDER_TYPE,   GLX_RGBA_BIT,
    GLX_DOUBLEBUFFER,  True,  // Request a double-buffered color buffer with 
    GLX_RED_SIZE,      1,     // the maximum number of bits per component    
    GLX_GREEN_SIZE,    1, 
    GLX_BLUE_SIZE,     1,
    None
  };
};

/* // There's no *****ng way to partition the variables separately into the header and cpp files
namespace glx_attr {
  // int* singleBufferAttributes;
  // int* doubleBufferAttributes;
};
*/



/** A Structure encapsulating an OpenGL %PBO (Pixel Buffer Object)
 * @ingroup openglthread_tag 
 */
struct PBO {
  GLuint   index;   ///< internal OpenGL/GPU index
  GLsizei  size;    ///< payload max size
  GLubyte* payload; ///< direct memory access (dma) memory address, returned by GPU
};


/** Encapsulates OpenGL %PBOs (Pixel Buffer Objects) for a YUV420 image
 * @ingroup openglthread_tag 
 */
class YUVPBO {
  
public:
  /** Default constructor
   * 
   * @param size Size of Y plane in bytes.  Size of U and V planes is calculated
   * \callgraph
   */
  YUVPBO(BitmapType bmtype);
  ~YUVPBO(); ///< Default destructor
  
private:
  void reserve();     ///< Reserve data on the GPU.  Used by the constructor only.
  
public:
  BitmapType bmtype;
  GLsizei    size;    ///< Size of Y plane in bytes.  Size of U and V planes is calculated
  
  GLuint   y_index;   ///< internal OpenGL/GPU index
  GLuint   u_index;   ///< internal OpenGL/GPU index
  GLuint   v_index;   ///< internal OpenGL/GPU index
  
  GLubyte* y_payload; ///< direct memory access (dma) memory address, returned by GPU
  GLubyte* u_payload; ///< direct memory access (dma) memory address, returned by GPU
  GLubyte* v_payload; ///< direct memory access (dma) memory address, returned by GPU
  
public:
  void upload(GLsizei isize, GLubyte* y, GLubyte* u, GLubyte* v); ///< Upload to GPU
  
};



/** A class encapsulating information about an OpenGL texture set (sizes, OpenGL reference ids, etc.)
 * @ingroup openglthread_tag 
 */
class TEX {
 
public:
  /** Default constructor
   * @param w pixmap width
   * @param h pixmap height
   * \callgraph
   */
  TEX(GLsizei w, GLsizei h);
  virtual ~TEX();            ///< Default virtual destructor
  
public: // format, dimensions
  GLint    internal_format;    ///< OpenGL internal format - this MUST be optimized!       
  GLint    format;             ///< OpenGL format of the texture - this MUST be optimized!
  GLsizei  w;         ///< Width of the largest plane (Y)
  GLsizei  h;         ///< Height of the largest plane (Y)
  
public: // OpenGL reference data: indices
  GLuint   index;     ///< OpenGL reference 
  
};

/** A class encapsulating information about an OpenGL texture set for a YUV pixmap (sizes, OpenGL reference ids, etc.)
 * @ingroup openglthread_tag 
 */
class YUVTEX : public TEX {
  
public:
  /** @copydoc TEX::TEX */
  YUVTEX(GLsizei w, GLsizei h); 
  ~YUVTEX(); ///< Default destructor
  
public:
  GLuint  y_index;       ///< internal OpenGL/GPU index referring to Y texture;
  GLuint  u_index;       ///< internal OpenGL/GPU index referring to U texture;
  GLuint  v_index;       ///< internal OpenGL/GPU index referring to V texture;
};


std::ostream &operator<<(std::ostream &os, YUVPBO const &m);
std::ostream &operator<<(std::ostream &os, YUVTEX const &m);


/** @ingroup openglthread_tag 
 *@{*/

int readbytes(const char* fname, char*& buffer); ///< Auxiliary routine for testing
int readyuvbytes(const char* fname, GLubyte*& Y, GLubyte*& U, GLubyte*& V); ///< Auxiliary routine for testing
void zeroyuvbytes(int ysize, GLubyte*& Y, GLubyte*& U, GLubyte*& V); ///< Auxiliary routine for testing

void getPBO(GLuint& index, GLsizei size, GLubyte*& payload); ///< Get PBO from the GPU.  There are two versions of this function: other one can be enabled in source for valgrind debugging
void releasePBO(GLuint* index, GLubyte* payload);

void getTEX(GLuint& index, GLint internal_format, GLint format, GLsizei w, GLsizei h); ///< Get texture from the GPU

void loadYUVPBO(YUVPBO* pbo, GLsizei size, GLubyte* y, GLubyte* u, GLubyte* v); ///< Load data from pointer addresses to PBOs
void loadYUVTEX(YUVPBO* pbo, YUVTEX* tex); ///< Load data from PBOs to textures

void peekYUVPBO(YUVPBO* pbo); ///< Print payload of a YUVPBO instance

/** @}*/


#endif