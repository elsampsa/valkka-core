#ifndef COMMON_HEADER_GUARD 
#define COMMON_HEADER_GUARD

/*
 * common.h : A list/recompilation of common header files
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
 *  @file    common.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.15.0 
 *  
 *  @brief List of common header files
 *
 */

// coding style:
// http://csweb.cs.wfu.edu/~fulp/CSC112/codeStyle.html
// doxygen:
// https://www.stack.nl/~dimitri/doxygen/manual/

#include <stdlib.h>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <string>

#include <iostream>
#include <fstream> // https://stackoverflow.com/questions/9816900/infile-incomplete-type-error
#include <iomanip>
#include <iterator>
#include <sstream>

#include <vector>  
#include <algorithm>
#include <sys/time.h>
#include <time.h>
// #include <linux/time.h>
// #include <sys/sysinfo.h>

#include <map>
#include <list>
#include <deque>

#include <chrono> 
#include <thread>

#include <sched.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>

#include <mutex>
#include <condition_variable>
#include <sys/eventfd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// usb cams & raw file write
#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

// OpenGL
#include<GL/glew.h>
#include<GL/glx.h>
#include<GL/glxext.h>

#define PY_ARRAY_UNIQUE_SYMBOL valkka_shmem_array_api

#endif


