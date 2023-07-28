#ifndef openglobject_HEADER_GUARD
#define openglobject_HEADER_GUARD
/*
 * openglobject.h : OpenGL objects, i.e. stuff that can be drawn on the OpenGL canvas on top the textures (boxes, etc.)
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
 *  @file    openglobject.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.5.0 
 *  
 *  @brief   OpenGL objects, i.e. stuff that can be drawn on the OpenGL canvas on top the textures (boxes, etc.)
 */ 

#include "common.h"
#include "shader.h"

/** A generic object that's drawn on top of the bitmaps.
 * 
 */
class OverlayObject {

public:
    OverlayObject();
    virtual ~OverlayObject();
    
protected: // opengl vaos etc.
    GLuint        VAO;     ///< id of the vertex array object
    GLuint        VBO;     ///< id of the vertex buffer object
    bool          is_set;  ///< have coordinates been given?
    
public:
    virtual void draw() = 0;
};

/** A rectangle that's drawn on top of the video bitmap.  Useful for object detection.
 * 
 */
class Rectangle : public OverlayObject {
    
public:
    Rectangle();
    virtual ~Rectangle();
    
public:
    virtual void draw();
    void setCoordinates(float left, float right, float top, float bottom);
    
protected:
    std::array<GLfloat,12> vertices;  ///< data of the vertex buffer object
};


#endif
