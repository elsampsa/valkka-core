/*
 * event.cpp :
 * 
 * Copyright 2023 Sampsa Riikonen
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
 *  @file    event.cpp
 *  @author  Sampsa Riikonen
 *  @date    2021
 *  @version 1.5.0 
 *  
 *  @brief 
 */ 

#include "event.h"

using namespace std::chrono; // seconds, milliseconds

Event::Event() : flag(false) {
}


Event::~Event() {

}


void Event::clear() {
    {
        std::unique_lock<std::mutex> lock(mutex);
        flag=false;
    }
    cv.notify_all();
}

void Event::set() {
    {
        std::unique_lock<std::mutex> lock(mutex);
        flag=true;
    }
    cv.notify_all();
}


bool Event::is_set() {
    return wait(0);
}

bool Event::wait(int timeout) 
{
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, milliseconds(timeout), [&]{return flag;});
}
