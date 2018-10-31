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
 *  @version 0.8.0 
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
        glBindVertexArray(VAO);
        glDrawArrays(GL_LINE_LOOP, 0, 4);
        glBindVertexArray(0);
    }
}
    
void Rectangle::setCoordinates(float left, float right, float top, float bottom) {
    vertices =std::array<GLfloat,12> {
        right, top,     0,
        right, bottom,  0,
        left,  bottom,  0,
        left,  top,     0
    };
    
    glBindVertexArray(VAO); // VAO works as a "mini program" .. we do all the steps below, when binding the VAO

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*vertices.size(), vertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    
    glBindVertexArray(0); // Unbind VAO
    
    is_set = true;
}
    
    
    
    

