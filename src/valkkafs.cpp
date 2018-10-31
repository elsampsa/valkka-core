/*
 * valkkafs.cpp :
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
 *  @file    valkkafs.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.8.0 
 *  
 *  @brief 
 */ 

#include "valkkafs.h"


ValkkaFS::ValkkaFS(const char *device_file, const char *block_file, long int blocksize, long int device_size) : device_file(device_file), block_file(block_file), blocksize(blocksize), device_size(device_size) {
}
    
ValkkaFS::~ValkkaFS() {
}


ValkkaFSWriterThread::ValkkaFSWriterThread(const char *name, ValkkaFS &valkkafs) : Thread(name), valkkafs(valkkafs) {
}
    
ValkkaFSWriterThread::~ValkkaFSWriterThread() {
}
    
 
ValkkaFSReaderThread::ValkkaFSReaderThread(const char *name, ValkkaFS &valkkafs) : Thread(name), valkkafs(valkkafs) {
}

ValkkaFSReaderThread::~ValkkaFSReaderThread() {
}


