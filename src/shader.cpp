/*
 * shader.cpp : OpenGL shaders for YUV to RGB interpolation
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
 *  @file    shader.cpp
 *  @author  Sampsa Riikonen
 *  @author  Markus Kaukonen
 *  @date    2017
 *  @version 0.17.5 
 *  
 *  @brief 
 *
 *  @section DESCRIPTION
 *  
 */ 

#include "shader.h"
#include "logging.h"


const char* Shader::vertex_shader_obj () { return
"#version 300 es\n"
"precision mediump float;\n"
"uniform mat4 transform;\n"
"layout (location = 0) in vec3 position;\n"
"void main()\n"
"{\n"
"  gl_Position = transform * vec4(position, 1.0f);\n"
"}\n";
}

const char* Shader::fragment_shader_obj () { return
"#version 300 es\n"
"precision mediump float;\n"
"out vec4 colour;\n"
"void main()\n"
"{\n"
"  colour = vec4(0,1,0,0);\n"
"}\n";
}



/*** RGB Shader Program ***/

const char* RGBShader::vertex_shader () { return
"#version 300 es\n"
"precision mediump float;\n"
// "in vec2 scaling;\n"
"uniform mat4 transform;\n"
"layout (location = 0) in vec3 position;\n"
"layout (location = 1) in vec2 texcoord;\n"

"out vec2 TexCoord;\n"

"void main()\n"
"{\n"
// "  gl_Position = vec4(position, 1.0f);\n"
// "  gl_Position = vec4(position, 1.0f) * vec4(scaling,1.0f,1.0f);\n"
"  gl_Position = transform * vec4(position, 1.0f);\n"
"  TexCoord = texcoord;\n"
"}\n";
}

const char* RGBShader::fragment_shader () { return
"#version 300 es\n"
"precision mediump float;\n"
"in vec2 TexCoord;\n"

"out vec4 color;\n"

"uniform sampler2D ourTexture;\n"

"void main()\n"
"{\n"
"  color = texture(ourTexture, TexCoord);\n"
"}\n";
}

/*** YUV Shader Program ***/

const char* YUVShader::vertex_shader () { return 
// shader vertex source code
// We swap the y-axis by substracing our coordinates from 1.
// This is done because most images have the top y-axis
// inversed with OpenGL's top y-axis.
// TexCoord = texcoord;
"#version 300 es\n"
"precision mediump float;\n"
// "in vec2 scaling;\n"
"uniform mat4 transform;\n"
"layout (location = 0) in vec3 position;\n"
"layout (location = 1) in vec2 texcoord;\n"
"out vec2 TexCoord;\n"
"void main()\n"
"{\n"
// "  gl_Position = vec4(position, 1.0f) * vec4(scaling,1.0f,1.0f);\n"
"  gl_Position = transform * vec4(position, 1.0f);\n"
"  //TexCoord = vec2(texcoord.x, 1.0 - texcoord.y);\n"
"  TexCoord = vec2(texcoord.x, texcoord.y);\n"
"}\n";
}

const char* YUVShader::fragment_shader  () { return
"#version 300 es\n"
"precision mediump float;\n"
"in vec3 ourColor;\n"
"in vec2 TexCoord;\n"
"uniform sampler2D texy; // Y \n"
"uniform sampler2D texu; // U \n"
"uniform sampler2D texv; // V \n"
"out vec4 colour;\n"
" // \n"
"vec3 yuv2rgb(in vec3 yuv) \n"
"{ \n"
"    // YUV offset  \n"
"    // const vec3 offset = vec3(-0.0625, -0.5, -0.5); \n"
"    const vec3 offset = vec3(-0.0625, -0.5, -0.5); \n"  
"    // RGB coefficients \n"
"    const vec3 Rcoeff = vec3( 1.164, 0.000,  1.596); \n"
"    const vec3 Gcoeff = vec3( 1.164, -0.391, -0.813); \n"
"    const vec3 Bcoeff = vec3( 1.164, 2.018,  0.000); \n"  
"    vec3 rgb; \n"
"    yuv = clamp(yuv, 0.0, 1.0); \n"
"    yuv += offset; \n"
"    rgb.r = dot(yuv, Rcoeff);  \n"
"    rgb.g = dot(yuv, Gcoeff); \n"  
"    rgb.b = dot(yuv, Bcoeff); \n"  
"    return rgb; \n"
"} \n"
" // \n"
"vec3 get_yuv_from_texture(in vec2 tcoord) \n"
"{ \n"
"    vec3 yuv; \n"
"    yuv.x = texture(texy, tcoord).r; \n"
"    // Get the U and V values \n"
"    yuv.y = texture(texu, tcoord).r; \n"
"    yuv.z = texture(texv, tcoord).r; \n"
"    return yuv; \n"
"} \n"
" // \n"
"vec4 mytexture2D(in vec2 tcoord) \n"
"{ \n"
"    vec3 rgb, yuv; \n"
"    yuv = get_yuv_from_texture(tcoord); \n"
"    // Do the color transform \n"
"    rgb = yuv2rgb(yuv); \n"
"    return vec4(rgb, 1.0); \n"
"} \n"
" // \n"
"void main()\n"
"{\n"
" // colour = texture(ourTexture1, TexCoord); \n"
" colour = mytexture2D(TexCoord); \n"
" // colour = texture(texy, TexCoord); \n" // debug (skip yuv=>rgb)
"}\n";
}





Shader::Shader() {
  /*
  compile(); // woops.. at constructor time, overwritten virtual methods are NOT called
  use();
  findVars();
  */
}

Shader::~Shader() {
#ifdef VALGRIND_GPU_DEBUG
#else
  glDeleteProgram(this->program);
#endif
}


void Shader::compile() {
  GLuint id_vertex_shader, id_fragment_shader, id_vertex_shader_obj, id_fragment_shader_obj;
  const char *source;
  int length, cc;
  GLint success;
  GLchar infoLog[512];
  
  // opengllogger.log(LogLevel::fatal) << 
  opengllogger.log(LogLevel::debug) << "Shader: compile: " <<std::endl;
  opengllogger.log(LogLevel::crazy) << "Shader: compile: vertex program=" << std::endl << vertex_shader() << std::endl;
  opengllogger.log(LogLevel::crazy) << "Shader: compile: fragment program=" << std::endl << fragment_shader() << std::endl;
  opengllogger.log(LogLevel::crazy) << "Shader: compile: fragment obj program=" << std::endl << fragment_shader_obj() << std::endl;
  
  // create and compiler vertex shader
  source=vertex_shader();
  id_vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  length = std::strlen(source);
  glShaderSource(id_vertex_shader, 1, &source, &length); 
  glCompileShader(id_vertex_shader);
  glGetShaderiv(id_vertex_shader, GL_COMPILE_STATUS, &success);
  if (!success)
  {
    glGetShaderInfoLog(id_vertex_shader, 512, NULL, infoLog);
    opengllogger.log(LogLevel::fatal) << "Shader: compile: vertex shader program (len="<<length<<") COMPILATION FAILED!" << std::endl << infoLog << std::endl;
  }

  // create and compile fragment shader
  source=fragment_shader();
  id_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  length = std::strlen(source);
  glShaderSource(id_fragment_shader, 1, &source, &length);   
  glCompileShader(id_fragment_shader);
  glGetShaderiv(id_fragment_shader, GL_COMPILE_STATUS, &success);
  if (!success)
  {
    glGetShaderInfoLog(id_fragment_shader, 512, NULL, infoLog);
    opengllogger.log(LogLevel::fatal) << "Shader: compile: fragment shader program (len="<<length<<") COMPILATION FAILED!" << std::endl << infoLog << std::endl;
  }

  // create and compiler vertex shader for overlay objects
  source=vertex_shader_obj();
  id_vertex_shader_obj = glCreateShader(GL_VERTEX_SHADER);
  length = std::strlen(source);
  glShaderSource(id_vertex_shader_obj, 1, &source, &length); 
  glCompileShader(id_vertex_shader_obj);
  glGetShaderiv(id_vertex_shader_obj, GL_COMPILE_STATUS, &success);
  if (!success)
  {
    glGetShaderInfoLog(id_vertex_shader_obj, 512, NULL, infoLog);
    opengllogger.log(LogLevel::fatal) << "Shader: compile: vertex shader program (len="<<length<<") COMPILATION FAILED!" << std::endl << infoLog << std::endl;
  }

  
  // create and compile fragment shader for overlay objects
  source=fragment_shader_obj();
  id_fragment_shader_obj = glCreateShader(GL_FRAGMENT_SHADER);
  length = std::strlen(source);
  glShaderSource(id_fragment_shader_obj, 1, &source, &length);   
  glCompileShader(id_fragment_shader_obj);
  glGetShaderiv(id_fragment_shader_obj, GL_COMPILE_STATUS, &success);
  if (!success)
  {
    glGetShaderInfoLog(id_fragment_shader_obj, 512, NULL, infoLog);
    opengllogger.log(LogLevel::fatal) << "Shader: compile: fragment shader obj program (len="<<length<<") COMPILATION FAILED!" << std::endl << infoLog << std::endl;
  }

  // Shader Program for bitmap interpolation
  this->program = glCreateProgram();
  opengllogger.log(LogLevel::debug) << "Shader: compile: program index=" << this->program << "\n";
  glAttachShader(this->program, id_vertex_shader);
  glAttachShader(this->program, id_fragment_shader);
  glLinkProgram(this->program);
  // Print linking errors if any
  glGetProgramiv(this->program, GL_LINK_STATUS, &success);
  if (!success)
  {
    glGetProgramInfoLog(this->program, 512, NULL, infoLog);
    opengllogger.log(LogLevel::fatal) << "Shader: compile: fragment shader LINKING FAILED!" << std::endl << infoLog << std::endl;
  }
  
  // Shader Program for geometric overlay objects
  this->program_obj = glCreateProgram();
  opengllogger.log(LogLevel::debug) << "Shader: compile: program_obj index=" << this->program << "\n";
  glAttachShader(this->program_obj, id_vertex_shader_obj);
  glAttachShader(this->program_obj, id_fragment_shader_obj);
  glLinkProgram(this->program_obj);
  // Print linking errors if any
  glGetProgramiv(this->program_obj, GL_LINK_STATUS, &success);
  if (!success)
  {
    glGetProgramInfoLog(this->program_obj, 512, NULL, infoLog);
    opengllogger.log(LogLevel::fatal) << "Shader: compile: fragment shader LINKING FAILED!" << std::endl << infoLog << std::endl;
  }
  
  // Delete the shaders as they're linked into our program now and no longer necessery
  glDeleteShader(id_vertex_shader);
  glDeleteShader(id_fragment_shader);
  glDeleteShader(id_vertex_shader_obj);
  glDeleteShader(id_fragment_shader_obj);
}


void Shader::findVars() {
  position=0; // this is hard-coded into the shader code (see "location=0")
  texcoord=1; // this is hard-coded into the shader code (see "location=1")
  object=0;   // hard-coded into the shader code for overlay objects ("location=0")
  
#ifdef VALGRIND_GPU_DEBUG
#else
  transform     =glGetUniformLocation(program,    "transform");
  transform_obj =glGetUniformLocation(program_obj,"transform");
#endif
  opengllogger.log(LogLevel::debug) << "Shader: findVars: Location of the transform matrix: " << transform << std::endl;
}


void Shader::scale(GLfloat fx, GLfloat fy) {
  GLfloat mat[4][4] = {
    {fx,               0.0f,             0.0f,   0.0f}, 
    {0.0f,             fy,               0.0f,   0.0f},
    {0.0f,             0.0f,             1.0f,   0.0f},
    {0.0f,             0.0f,             0.0f,   1.0f}
  };
#ifdef VALGRIND_GPU_DEBUG
#else
  glUniformMatrix4fv(transform, 1, GL_FALSE, mat[0]);
#endif
}


void Shader::use() {
  // opengllogger.log(LogLevel::crazy) << "Shader: use: using program index=" << this->program << std::endl;
#ifdef VALGRIND_GPU_DEBUG
#else
  glUseProgram(this->program);
#endif
}

void Shader::use_obj() {
  // opengllogger.log(LogLevel::crazy) << "Shader: use: using program index=" << this->program << std::endl;
#ifdef VALGRIND_GPU_DEBUG
#else
  glUseProgram(this->program_obj);
#endif
}

void Shader::validate() {
#ifdef VALGRIND_GPU_DEBUG
#else
  GLint params, maxLength;
  //The maxLength includes the NULL character
  // std::vector<GLchar> infoLog(maxLength);
  
  std::cout << std::endl << "Shader: validating program index=" << program << std::endl;
  std::cout              << "Shader: is program              =" << bool(glIsProgram(program)) << std::endl;
  glValidateProgram(program);
  glGetProgramiv(program,GL_VALIDATE_STATUS,&params);
  std::cout              << "Shader: validate status         =" << params << std::endl;
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);  
  char infoLog[maxLength];

  glGetProgramInfoLog(program, maxLength, &maxLength, &infoLog[0]);
  std::cout              << "Shader: infoLog length          =" << maxLength << std::endl;
  std::cout              << "Shader: infoLog                 =" << std::string(infoLog) << std::endl;
  std::cout << std::endl;
#endif
}



RGBShader::RGBShader() : Shader() {
#ifdef VALGRIND_GPU_DEBUG
#else
  compile();
  use();
  findVars();
#endif
}


RGBShader::~RGBShader() {
}



YUVShader::YUVShader() : Shader() {
#ifdef VALGRIND_GPU_DEBUG
#else
  compile();
  use();
  findVars();
#endif
}

YUVShader::~YUVShader() {
}


void YUVShader::findVars() {
  position=0; // this is hard-coded into the shader code (see "location=0")
  texcoord=1; // this is hard-coded into the shader code (see "location=1")
  object=0;
  
  opengllogger.log(LogLevel::debug) << "YUVShader: findVars: Location of position: " << position << std::endl;
  opengllogger.log(LogLevel::debug) << "YUVShader: findVars: Location of texcoord: " << texcoord << std::endl;
  
  transform=glGetUniformLocation(program,"transform");
  opengllogger.log(LogLevel::debug) << "YUVShader: findVars: Location of the transform matrix: " << transform << std::endl;
  
  transform_obj=glGetUniformLocation(program_obj,"transform");
  opengllogger.log(LogLevel::debug) << "YUVShader: findVars: Location of the transform matrix in obj: " << transform_obj << std::endl;
  
  texy=glGetUniformLocation(program,"texy");
  opengllogger.log(LogLevel::debug) << "YUVShader: findVars: Location of texy: " << texy << std::endl;
  
  texu=glGetUniformLocation(program,"texu");
  opengllogger.log(LogLevel::debug) << "YUVShader: findVars: Location of texu: " << texu << std::endl;
  
  texv=glGetUniformLocation(program,"texv");
  opengllogger.log(LogLevel::debug) << "YUVShader: findVars: Location of texv: " << texv << std::endl;
}


