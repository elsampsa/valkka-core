/*
 * rawrite.cpp : Write directly to files and devices with POSIX O_DIRECT
 * 
 * Copyright 2017, 2018 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * Authors: Petri Eranko <petri.eranko@dasys.fi>
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
 *  @file    rawrite.cpp
 *  @author  Petri Eranko
 *  @date    2019
 *  @version 1.0.0 
 *  
 *  @brief 
 */ 

#include "rawrite.h"


RaWriter::RaWriter(const char* filename, bool direct) : filename(filename), fd(0), count(0) {
    if (direct) {
        fd = open(this->filename.c_str(), O_CREAT | O_WRONLY | O_LARGEFILE | O_NONBLOCK | O_DIRECT, S_IRWXU);
        // fd = open(this->filename.c_str(), O_CREAT | O_WRONLY | O_DIRECT, S_IRWXU); // no f-way to make this work
    }
    else {
        // fd = open(this->filename.c_str(), O_CREAT | O_WRONLY | O_LARGEFILE | O_NONBLOCK | O_DSYNC, S_IRWXU);
        fd = open(this->filename.c_str(), O_CREAT | O_WRONLY | O_LARGEFILE | O_NONBLOCK, S_IRWXU);
        // fd = open(this->filename.c_str(), O_CREAT | O_WRONLY, S_IRWXU);
    }
    
    // std::cout << "RaWriter: fd = " << fd << std::endl;
    // tmp = new char[FS_GRAIN_SIZE];
    
    is_open = true;
    if (fd < 0) {
        // http://man7.org/linux/man-pages/man3/errno.3.html
        std::cout << "RaWriter: open file failed with " << errno << " for file " << this->filename << std::endl;
        is_open = false;
    }
    else {
        seek(0);
    }
}


RaWriter::~RaWriter() {
    if (is_open) {
        close_();
    }
}


bool RaWriter::writeGrain() {
    std::size_t i;
    // std::cout << "RaWriter: writeGrain" << std::endl;
    
    if (!is_open) {
        return false;
    }
    // std::cout << "RaWriter: writeGrain: writing to " << lseek(fd, 0, SEEK_CUR) << std::endl;
    i = write(fd, ptmp, FS_GRAIN_SIZE);
    if (int(i) < 0) {
        std::cout << "RaWriter: error: write failed with " << errno << std::endl;
        close_();
        return false;
    }
    count = 0;
    return true;
}


void RaWriter::dump(const char* source, std::size_t len) {
    std::size_t sourcecount = 0; // source counter
    std::size_t i;
    if (!is_open) {
        return;
    }
    while (sourcecount < len) { // BYTES LEFT
        tmp[count] = source[sourcecount];
        sourcecount++;
        count++;
        if (count >= FS_GRAIN_SIZE) { // FLUSH
            writeGrain();
            // after the first grain, try to write without memcopy
            while ((sourcecount+FS_GRAIN_SIZE) < len) { // next FS_GRAIN_SIZE bytes can be taken from the source directly
                i = write(fd, (const void*)&source[sourcecount], FS_GRAIN_SIZE);
                if (int(i) < 0) {
                    std::cout << "RaWriter: error: write failed with " << errno << std::endl;
                    close_();
                    return;
                }
                sourcecount = sourcecount + FS_GRAIN_SIZE;
            }
        } // FLUSH
    } // BYTES LEFT
}


void RaWriter::fill(std::size_t len) {
    std::size_t sourcecount = 0; // source counter
    std::size_t i;
    int c;
    if (!is_open) {
        return;
    }
    while (sourcecount < len) { // BYTES LEFT
        tmp[count] = 0;
        sourcecount++;
        count++;
        if (count >= FS_GRAIN_SIZE) { // FLUSH
            writeGrain();
            // after the first flush, try to write without memcopy
            memset(ptmp, 0, FS_GRAIN_SIZE);
            while ((sourcecount + FS_GRAIN_SIZE) < len) { // next FS_GRAIN_SIZE bytes can be taken from the tmp directly
                i = write(fd, ptmp, FS_GRAIN_SIZE);
                if (int(i) < 0) {
                    std::cout << "RaWriter: error: write failed with " << errno << std::endl;
                    close_();
                    return;
                }
                sourcecount = sourcecount + FS_GRAIN_SIZE;
            }
        } // FLUSH
    } // BYTES LEFT
}


void RaWriter::finish() {
    std::size_t i;
    if (!is_open) {
        return;
    }
    if (count >= FS_GRAIN_SIZE) {
        return;
    }
    while (count < FS_GRAIN_SIZE) {
        tmp[count] = 0;
        count++;
    }
    writeGrain();
}
    

void RaWriter::fwd(off_t len) {
    off_t res = lseek(fd, len, SEEK_CUR);
    // std::cout << "RaWriter: fwd: req, res: " << len << " " << res << std::endl;
    count = 0; // resets buffering
}


void RaWriter::seek(off_t len) {
    off_t res = lseek(fd, len, SEEK_SET);
    // std::cout << "RaWriter: seek: req, res: " << len << " " << res << std::endl;
    count = 0; // resets buffering
}


off_t RaWriter::getPos() {
    return lseek(fd, 0, SEEK_CUR) + count;
}


int RaWriter::getCount() {
    return count;
}


void RaWriter::close_() {
    finish();
    close(fd);
    is_open = false;
}



RawReader::RawReader(const char* filename, bool direct) : filename(filename), fd(0), count(0) {
    if (direct) {
        fd = open(this->filename.c_str(), O_RDONLY | O_LARGEFILE | O_NONBLOCK | O_DIRECT);
    }
    else {
        fd = open(this->filename.c_str(), O_RDONLY | O_LARGEFILE | O_NONBLOCK);
        // fd = open(this->filename.c_str(), O_RDONLY | O_LARGEFILE | O_NONBLOCK | O_DSYNC);
        // fd = open(this->filename.c_str(), O_CREAT | O_WRONLY);
    }    
    is_open = true;
    if (fd < 0) {
        std::cout << "RawReader: open file failed with " << errno <<" for file " << this->filename << std::endl;
        is_open = false;
    }
    else {
        seek(0);
    }
}


bool RawReader::readGrain() {
    std::size_t i;
    // std::cout << "RawReader: readGrain" << std::endl;
    
    if (!is_open) {
        return false;
    }
    i = read(fd, ptmp, FS_GRAIN_SIZE);
    if (int(i) < 0) {
        std::cout << "RaWriter: error: read failed with " << errno << std::endl;
        close_();
        return false;
    }
    count = 0; // number of consumed bytes
    return true;
}


RawReader::~RawReader() {
    if (is_open) {
        close_();
    }
}

void RawReader::get(char* buf, std::size_t len) {
    std::size_t targetcount = 0;
    int i;
    if (count < 1) {
        readGrain();
    }
    while (targetcount < len) { // BYTES REQUIRED
        // extract bytes from tmp
        buf[targetcount] = tmp[count];
        targetcount++;
        count++;
        if (count >= FS_GRAIN_SIZE) { // time for the next grain .. // NEXT GRAIN
            count = 0;
            while ( (targetcount + FS_GRAIN_SIZE) < len ) { // read consecutive grains without extra caching if possible
                // std::cout << "RawReader: read whole grain" << std::endl;
                i = read(fd, (void*)&buf[targetcount], FS_GRAIN_SIZE);
                targetcount = targetcount + FS_GRAIN_SIZE;
            }
            if (targetcount < len) { // there's still something for this variable ..
                readGrain(); // cache the remaining bytes and start pulling bytes again from next grain
            }
        } // NEXT GRAIN
    } // BYTES REQUIRED
}

    
void RawReader::fwd(off_t len) {
    off_t res = lseek(fd, len, SEEK_CUR);
    // std::cout << "RawReader: fwd: req, res: " << len << " " << res << std::endl;
    count = 0; // resets buffering
}
    
void RawReader::seek(off_t len) {
    off_t res = lseek(fd, len, SEEK_SET);
    // std::cout << "RawReader: seek: req, res: " << len << " " << res << std::endl;
    count = 0; // resets buffering
}
    
void RawReader::close_() {
    close(fd);
    is_open = false;
    count = 0;
}









