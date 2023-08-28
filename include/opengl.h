#ifndef OPENGL_HEADER_GUARD 
#define OPENGL_HEADER_GUARD

/*
 * opengl.h : OpenGL calls for reserving PBOs and TEXtures, plus some auxiliary routines
 * 
 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>
 *
 */

/** 
 *  @file    opengl.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.5.3 
 *  
 *  @brief OpenGL calls for reserving PBOs and TEXtures, plus some auxiliary routines
 *  
 */ 

#include "frame.h"
#include "constant.h"
//#include<GL/glew.h>
//#include<GL/glx.h>
//#include<GL/glxext.h>


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


/** @ingroup openglthread_tag 
 *@{*/

int readbytes(const char* fname, char*& buffer); ///< Auxiliary routine for testing
int readyuvbytes(const char* fname, GLubyte*& Y, GLubyte*& U, GLubyte*& V); ///< Auxiliary routine for testing
void zeroyuvbytes(int ysize, GLubyte*& Y, GLubyte*& U, GLubyte*& V); ///< Auxiliary routine for testing

void getPBO(GLuint& index, GLsizei size, GLubyte*& payload); ///< Get PBO from the GPU.  There are two versions of this function: other one can be enabled in source for valgrind debugging
void releasePBO(GLuint* index, GLubyte* payload);

void getTEX(GLuint& index, GLint internal_format, GLint format, GLsizei w, GLsizei h); ///< Get texture from the GPU

/** @}*/


#endif
