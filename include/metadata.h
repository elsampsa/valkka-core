#ifndef metadata_HEADER_GUARD
#define metadata_HEADER_GUARD
/*
 * metadata.h : enums and structs for converting byte blobs into metadata
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
 *  @file    metadata.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.6.1 
 *  
 *  @brief   libValkka pre-reserves all frames, both byte payload and metadata payload.
 *           After memory reservation, a frame can change its internal types, namely codec 
 *           and metadata related to that codec.  So we need typecast the correct
 *           metadata from the pre-reserved metadata byte blobs
 */ 

#endif

#include "constant.h"

static const int METADATA_MAX_SIZE = 10*1024; // 10 kB


enum class MuxMetaType
{
    none, ///< unknown
    fragmp4
};


struct FragMP4Meta {                                  // <pyapi>
    FragMP4Meta() : is_first(false),                  // <pyapi> 
    size(0), slot(0), mstimestamp(0) {}               // <pyapi>
    char name[4];                                     // <pyapi>
    bool is_first;                                    // <pyapi>
    std::size_t size; ///< Actual size copied         // <pyapi>
    SlotNumber slot;                                  // <pyapi>
    long int mstimestamp;                             // <pyapi>
};                                                    // <pyapi>


/* test code:
int main(int argv, char** argvc) {
    std::vector<uint8_t> meta_blob;
    meta_blob.reserve(METADATA_MAX_SIZE);
    FragMP4Meta* metap;
    metap = (FragMP4Meta*)(meta_blob.data());
    memcpy(&metap->name[0], "moof", 4);
    metap->is_first = true;

    std::cout << std::string(metap->name) << std::endl;
    std::cout << int(metap->is_first) << std::endl;
}
*/
