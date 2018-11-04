#ifndef usbthread_HEADER_GUARD
#define usbthread_HEADER_GUARD
/*
 * usbthread.h : USB Camera control and streaming
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
 *  @file    usbthread.h
 *  @author  Sampsa Riikonen
 *  @date    2018
 *  @version 0.1
 *  
 *  @brief   USB Camera control and streaming
 */ 


#include "common.h"

#define CLEAR(x) memset(&(x), 0, sizeof(x))

// at the python level, check this directory: /sys/class/video4linux
// .. that directory has video0/ video1/   .. with file "name", etc.
// .. /dev/video0 has the actualy device for reading

int xioctl(int fh, int request, void *arg);

enum v4l_status {
    none,           ///< undefined (initial value)
    not_found,      ///< file not found
    not_device,     ///< not device file
    not_read,       ///< could not read device
    not_v4l2,       ///< not v4l2 device
    not_video_cap,  ///< not video capture devices
    not_io,         ///< does not support file io (not used anyway)
    not_stream,     ///< does not support streaming
    not_ptr,        ///< does not support user pointers
    ok_open,        ///< streaming device ok
    not_format,     ///< could not achieve the requested format
    ok_format       ///< everything's ok!
};


class V4LDevice {
    
public:
    
    /** Default constructor
     * 
     * Open the device, query its capabilities
     * 
     * - Sets opening state
     * 
     * 
     */
    
    V4LDevice(const char *dev);
    ~V4LDevice();
    
protected:
    std::string     dev;     ///< Name of the device
    v4l_status      status;  ///< State of the device
    int             fd;      ///< File number
    v4l2_capability cap;
    v4l2_format     fmt;
    
    
public: // getters
    const v4l_status        getStatus() {return this->status;} 
    const v4l2_capability   getCapability() {return this->cap;}
    
    
public:
    void request(int width, int height, int pix_fmt);
    void initStreaming();
    
};





#endif
