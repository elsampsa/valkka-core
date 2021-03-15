#ifndef alsa_HEADER_GUARD
#define alsa_HEADER_GUARD
/*
 * alsa.h :
 * 
 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
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
 *  @file    alsa.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
 *  
 *  @brief
 */ 


struct ALSASinkContext {
    std::string name;
    int cardindex;
};


class ALSASink {
    // has a state:
    // can be initialized or not..
    // format can be changed on-the-fly
    // but this requires that underlying alsa
    // instances are re-created..

public:
    ALSASink(ALSASinkContext ctx);
    virtual ~ALSASink();

public:
    push(SoundFrame* f);

};


struct ALSASourceContext {
    std::string name;
    int cardindex;
};

class ALSASource {

public:
    ALSASource(ALSASourceContext ctx);
    virtual ~ALSASource();
};



#endif
