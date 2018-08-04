#ifndef TEX_HEADER_GUARD 
#define TEX_HEADER_GUARD

/*
 * tex.h : Handling OpenGL textures.
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
 *  @file    tex.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.5.2 
 *  
 *  @brief Handling OpenGL textures.
 *  
 */ 

#include "frame.h"
#include "opengl.h"
#include "constant.h"
#include "macro.h"


/** A class encapsulating information about an OpenGL texture set (sizes, OpenGL reference ids, etc.)
 * @ingroup openglthread_tag 
 */
class TEX {
 
public:
  /** Default constructor
   * @param width pixmap width
   * @param height pixmap height
   * \callgraph
   */
  TEX(BitmapPars bmpars);        ///< Default ctor
  virtual ~TEX();                ///< Default virtual destructor
  ban_copy_ctor(TEX);
  ban_copy_asm(TEX);


public: // format, dimensions
  BitmapPars bmpars;             ///< Max bitmap dimensions (pre-reserved)
  // BitmapPars source_bmpars;      ///< Bitmap true dimensions
  GLint      internal_format;    ///< OpenGL internal format 
  GLint      format;             ///< OpenGL format of the texture
  
public: // OpenGL reference data: indices
  GLuint   index;     ///< OpenGL reference 
};



/** A class encapsulating information about an OpenGL texture set for a YUV pixmap (sizes, OpenGL reference ids, etc.)
 * 
 * @ingroup openglthread_tag 
 */
class YUVTEX : public TEX {
  
public:
  /** @copydoc TEX::TEX */
  YUVTEX(BitmapPars bmpars); 
  ~YUVTEX(); ///< Default destructor
  YUVTEX(const YUVTEX &f) : TEX(f) notice_ban_copy_ctor();
  //ban_copy_ctor(YUVTEX); // must call TEX in the init list ..
  ban_copy_asm(YUVTEX);
  
public:
  GLuint  y_index;       ///< internal OpenGL/GPU index referring to Y texture;
  GLuint  u_index;       ///< internal OpenGL/GPU index referring to U texture;
  GLuint  v_index;       ///< internal OpenGL/GPU index referring to V texture;
  
public:
  void loadYUV(const GLubyte* Y, const GLubyte* U, const GLubyte* V); ///< Load YUV from memory buffers
  void loadYUVFrame(YUVFrame *yuvframe);            ///< Transfer from YUVFrame's PBO's to textures
};

std::ostream &operator<<(std::ostream &os, YUVTEX const &m);

#endif

