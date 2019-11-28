#ifndef rawrite_HEADER_GUARD
#define rawrite_HEADER_GUARD
/*
 * rawrite.h : Write directly to files and devices with POSIX O_DIRECT
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
 *  @file    rawrite.h
 *  @author  Petri Eranko
 *  @date    2019
 *  @version 0.14.1 
 *  
 *  @brief   Write directly to files and devices with POSIX O_DIRECT
 */ 


/*
 *  So, why this?
 * 
 *  - We want to avoid excessive caching, as we don't need it
 *  - We would even like to use DMA writes to a block device (fdd/sdd)
 *  - In the "pure-cpp" way, we should subclass fstream to support our own custom streambuffer class
 *    https://stackoverflow.com/questions/12997131/stdfstream-buffering-vs-manual-buffering-why-10x-gain-with-manual-buffering
 *    example of a custom streambuf: 
 *    https://stackoverflow.com/questions/42647267/subclass-fstream-and-decrypt-data-on-the-fly?rq=1
 *  - And also subclass the fstream class itself.  This would be quite tricky. 
 * 
 *  In order to use the linux O_DIRECT, DMA (Direct Memory Acces) to the hdd / sdd must be enabled.  This might be tricky:
 * 
 *  https://stackoverflow.com/questions/10512987/o-direct-flag-not-working
 *  https://www.tldp.org/HOWTO/archived/Ultra-DMA/Ultra-DMA-8.html
 *  https://www.linuxquestions.org/questions/linux-hardware-18/hdparm-d1-dev-hda-gives-me-hdio_set_dma-failed-operation-not-permitted-260894/
 *  https://www.linuxquestions.org/questions/debian-26/checking-the-dma-udma-modes-for-sata-drive-589717/
 * 
 *  Low-level file I/O in general in Linux:
 *  
 *  http://man7.org/linux/man-pages/man2/open.2.html
 *  http://man7.org/linux/man-pages/man3/errno.3.html
 *  http://man7.org/linux/man-pages/man2/close.2.html
 *  http://man7.org/linux/man-pages/man2/lseek.2.html
 * 
 */

#include "common.h"
#include "constant.h"

class RaWriter {
    
public:
    RaWriter(const char* filename, bool direct = false);
    ~RaWriter();
    
protected:
    std::string filename;
    int fd;
    int count;
    char tmp[FS_GRAIN_SIZE]; // in'da'stack
    void *ptmp = &tmp[0]; // alias
    // char* tmp;
    bool is_open;
    
public:
    bool writeGrain();
    void dump(const char* buf, std::size_t len);    ///< write with buffering.  flush occurs at grain boundaries
    // void dump_(const char* buf, std::size_t len);   ///< write without buffering and with immediate flush
    void fill(std::size_t len);                     ///< write null bytes
    void finish();                                  ///< write null bytes up to the end of grain
    void fwd(off_t len);
    void seek(off_t len);
    off_t getPos();
    int getCount();
    void close_();
};


class RawReader {
    
public:
    RawReader(const char* filename, bool direct = false);
    ~RawReader();
    
protected:
    std::string filename;
    int fd;
    int count;
    char tmp[FS_GRAIN_SIZE]; // in'da'stack
    void *ptmp = &tmp[0]; // alias
    // char* tmp;
    bool is_open;
    
public:
    bool readGrain();
    void get(char* buf, std::size_t len);
    void fwd(off_t len);
    void seek(off_t len);
    void close_();
};





#endif
