#ifndef event_HEADER_GUARD
#define event_HEADER_GUARD
/*
 * event.h :
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
 *  @file    event.h
 *  @author  Sampsa Riikonen
 *  @date    2023
 *  @version 1.5.2 
 *  
 *  @brief
 */ 
#include <mutex>
#include <condition_variable>

/**Python-like threading/multiprocessing.Event class
 * 
 * 
 */
class Event { 

public:
    Event();
    ~Event();

public:
    void clear();
    void set();
    bool is_set();
    bool wait(int timeout = 0);

private:
    std::mutex mutex;
    std::condition_variable cv;
    bool flag;
};
#endif
