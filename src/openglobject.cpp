/*
 * openglobject.cpp :
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
 *  @file    openglobject.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.17.5 
 *  
 *  @brief 
 */ 

#include "openglobject.h"
#include "logging.h"


OverlayObject::OverlayObject() : is_set(false) {
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
}
    
    
OverlayObject::~OverlayObject() {
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
}
    
    
Rectangle::Rectangle() : OverlayObject() {
}

Rectangle::~Rectangle() {
}

void Rectangle::draw() {
    // opengllogger.log(LogLevel::debug) << "Rectangle: draw: is_set = " << is_set << std::endl;
    if (is_set) {
        // shader->use();
        glBindVertexArray(VAO);
        glDrawArrays(GL_LINE_LOOP, 0, 4);
        glBindVertexArray(0);
    }
}
    
void Rectangle::setCoordinates(float left, float right, float top, float bottom) {
    
    // let's go from 0..1 coordinates to -1..1 coordinates, i.e.
    // transform with (x * 2 - 1)
    
    vertices =std::array<GLfloat,12> {
        right*(GLfloat)2.-(GLfloat)1., top*   (GLfloat)2.-(GLfloat)1.,  0,
        right*(GLfloat)2.-(GLfloat)1., bottom*(GLfloat)2.-(GLfloat)1.,  0,
        left* (GLfloat)2.-(GLfloat)1., bottom*(GLfloat)2.-(GLfloat)1.,  0,
        left* (GLfloat)2.-(GLfloat)1., top*   (GLfloat)2.-(GLfloat)1.,  0
    };
    
    /*
    color = std::array<GLfloat, 4> {
        GLfloat(0), GLfloat(1), GLfloat(0), GLfloat(0)  //RGBA
    };
    */
    
    glBindVertexArray(VAO); // VAO works as a "mini program" .. we do all the steps below, when binding the VAO

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*vertices.size(), vertices.data(), GL_STATIC_DRAW);

    // color attribute
    //glVertexAttribPointer(shader->color_rgba, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid*)0);
    //glEnableVertexAttribArray(shader->color_rgba); // this refers to (location=0) in the shader program
    
    glEnableVertexAttribArray(0); // position in the shader program
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    
    glBindVertexArray(0); // Unbind VAO
    
    is_set = true;
}
    
    
    
    

