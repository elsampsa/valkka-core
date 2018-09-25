#ifndef SHADERS_HEADER_GUARD 
#define SHADERS_HEADER_GUARD
/*
 * shaders.h : OpenGL shaders for YUV to RGB interpolation
 * 
 * Copyright 2017, 2018 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi> and Markus Kaukonen <markus.kaukonen@iki.fi>
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
 *  @file    shaders.h
 *  @author  Sampsa Riikonen
 *  @author  Markus Kaukonen
 *  @date    2017
 *  @version 0.5.4 
 *  
 *  @brief OpenGL shaders for YUV to RGB interpolation
 *  
 */ 

// #include<GL/glew.h>
// #include<GL/glx.h>
#include "common.h"
#include "macro.h"

/** A general purpose shader class.  Subclass for, say:
 * 
 * - RGB interpolation
 * - YUV interpolation 
 * - YUV interpolation and Fisheye projection
 * - etc.
 * 
 * 
 */
class Shader {

public:
  /** Default constructor.  Calls Shader::compile and Shader::findVars
   */
  Shader();
  virtual ~Shader(); ///< Default destructor
  ban_copy_ctor(Shader);
  ban_copy_asm(Shader);

protected: // functions that return shader programs
  virtual const char* vertex_shader()     =0;
  virtual const char* fragment_shader()   =0;
  
  
public: // declare GLint variable references here with "* SHADER PROGRAM VAR"
  GLint  transform; ///< OpenGL VERTEX SHADER PROGRAM VAR : transformation matrix
  GLint  position;  ///< OpenGL VERTEX SHADER PROGRAM VAR : position vertex array.  Typically "hard-coded" into the shader code with (location=0)
  GLint  texcoord;  ///< OpenGL VERTEX SHADER PROGRAM VAR : texture coordinate array. Typically "hard-coded" into the shader code with (location=1)
  
protected:
  GLuint program;   ///< OpenGL reference to shader program

public:
  void compile();   ///< Compile shader
  void virtual findVars();  ///< Link shader program variable references to the shader program
  void scale(GLfloat fx, GLfloat fy); ///< Set transformation matrix to simple scaling
  void use();       ///< Use this shader program
  void validate();  ///< Validate shader program
  
};



class RGBShader : public Shader {
  
public:
  RGBShader();
  ~RGBShader();
  ban_copy_ctor(RGBShader);
  ban_copy_asm(RGBShader);

protected: // functions that return shader programs
  const char* vertex_shader();
  const char* fragment_shader();

};



class YUVShader : public Shader {

public:
  YUVShader();
  ~YUVShader();
  ban_copy_ctor(YUVShader);
  ban_copy_asm(YUVShader);
  
public: // declare GLint variable references here with "* SHADER PROGRAM VAR"
  GLint  texy; ///< OpenGL VERTEX SHADER PROGRAM VAR : Y texture
  GLint  texu; ///< OpenGL VERTEX SHADER PROGRAM VAR : U texture
  GLint  texv; ///< OpenGL VERTEX SHADER PROGRAM VAR : V texture
  
protected: // functions that return shader programs
  const char* vertex_shader();
  const char* fragment_shader();
  
public: 
  void findVars();
  
};

#endif
