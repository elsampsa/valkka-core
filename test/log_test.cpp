/*
 * log_test.cpp : Test the loggin utility
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
 *  @file    log_test.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.3.5 
 *  
 *  @brief Test logging utility
 *
 */ 

#include "logging.h"

int main(int argc, char** argcv) {

// Logger logger;
// logger=Logger();
  
framelogger.setLevel(LogLevel::normal);  
framelogger.log(LogLevel::debug) << "kikkelis kokkelis 1" << std::endl;

framelogger.setLevel(LogLevel::debug);
framelogger.log(LogLevel::debug) << "kikkelis kokkelis 2" << std::endl;

framelogger.setLevel(LogLevel::normal);

}


