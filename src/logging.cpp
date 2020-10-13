/*
 * logging.cpp : Logging with various log levels
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
 *  @file    logging.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.0.1 
 *  @brief   Logging with various log levels
 */ 

#include "logging.h"

Logger::Logger(int log_level) : log_level(log_level), current_level(log_level) {};
/*
template <typename T> // T refers to *any* type
Logger& Logger::operator<<(const T& x)
{
  // std::cout << "current_level, log_level=" << current_level << " " << log_level << std::endl;
  // if (current_level<=log_level or current_special_level==special_level) {
  if (current_level<=log_level) {
    std::cout << x;
  }
  return *this;
}
*/

void Logger::setLevel(int level) {
  // std::cout << "Logger: setLevel: setting level to "<<level<<std::endl;
  log_level=level; 
}

Logger& Logger::log(int level) {
  current_level=level;
  return *this;
}

// take in a function with the custom signature
Logger& Logger::operator<<(LoggerManipulator manip) // .. when we encounter that function that eats Logger
{
  // call the function, and return it's value
  return manip(*this);
}

// define the custom endl for this stream.
// note how it matches the `LoggerManipulator`
// function signature
Logger& Logger::endl(Logger& stream)
{
  // print a new line
  std::cout << std::endl;
  // do other stuff with the stream
  // std::cout, for example, will flush the stream
  stream << "Called Logger::endl!" << std::endl;
  return stream;
}

// define an operator<< to take in std::endl
Logger& Logger::operator<<(StandardEndLine manip)
{
  // call the function, but we cannot return it's value
  // if (current_level<=log_level or current_special_level==special_level) {
  // std::cout << "Logger: log_level=" << log_level << std::endl;
  
  if (current_level<=log_level) {
    manip(std::cout);
  }
  return *this;
}


Logger framelogger;
Logger filterlogger;
Logger livelogger;
Logger threadlogger;
Logger livethreadlogger;
Logger avthreadlogger;
Logger decoderlogger;
Logger fifologger;
Logger opengllogger;
Logger filelogger;
Logger filethreadlogger;
Logger usblogger;
Logger valkkafslogger;

extern void setLogLevel_framelogger(int level)      { framelogger.setLevel(level); };
extern void setLogLevel_filterlogger(int level)     { filterlogger.setLevel(level); };     
extern void setLogLevel_livelogger(int level)       { livelogger.setLevel(level); };    
extern void setLogLevel_threadlogger(int level)     { threadlogger.setLevel(level); };     
extern void setLogLevel_livethreadlogger(int level) { livethreadlogger.setLevel(level); };    
extern void setLogLevel_avthreadlogger(int level)   { avthreadlogger.setLevel(level); };     
extern void setLogLevel_decoderlogger(int level)    { decoderlogger.setLevel(level); };     
extern void setLogLevel_fifologger(int level)       { fifologger.setLevel(level); };     
extern void setLogLevel_opengllogger(int level)     { opengllogger.setLevel(level); };     
extern void setLogLevel_filelogger(int level)       { filelogger.setLevel(level); };     
extern void setLogLevel_filethreadlogger(int level) { filethreadlogger.setLevel(level); };
extern void setLogLevel_usblogger(int level)        { usblogger.setLevel(level); };
extern void setLogLevel_valkkafslogger(int level)   { valkkafslogger.setLevel(level); };


void crazy_log_all() {
  framelogger         .setLevel(LogLevel::crazy);
  filterlogger        .setLevel(LogLevel::crazy);
  livelogger          .setLevel(LogLevel::crazy);
  threadlogger        .setLevel(LogLevel::crazy);
  livethreadlogger    .setLevel(LogLevel::crazy);
  avthreadlogger      .setLevel(LogLevel::crazy);
  decoderlogger       .setLevel(LogLevel::crazy);
  fifologger          .setLevel(LogLevel::crazy);
  opengllogger        .setLevel(LogLevel::crazy);
  filelogger          .setLevel(LogLevel::crazy);
  filethreadlogger    .setLevel(LogLevel::crazy);
  usblogger    .setLevel(LogLevel::crazy);
  valkkafslogger    .setLevel(LogLevel::crazy);
}


void debug_log_all() {
  framelogger         .setLevel(LogLevel::debug);
  filterlogger        .setLevel(LogLevel::debug);
  livelogger          .setLevel(LogLevel::debug);
  threadlogger        .setLevel(LogLevel::debug);
  livethreadlogger    .setLevel(LogLevel::debug);
  avthreadlogger      .setLevel(LogLevel::debug);
  decoderlogger       .setLevel(LogLevel::debug);
  fifologger          .setLevel(LogLevel::debug);
  opengllogger        .setLevel(LogLevel::debug);
  filelogger          .setLevel(LogLevel::debug);
  filethreadlogger    .setLevel(LogLevel::debug);
  usblogger    .setLevel(LogLevel::debug);
  valkkafslogger    .setLevel(LogLevel::debug);
}

void normal_log_all() {
  framelogger         .setLevel(LogLevel::normal);
  filterlogger        .setLevel(LogLevel::normal);
  livelogger          .setLevel(LogLevel::normal);
  threadlogger        .setLevel(LogLevel::normal);
  livethreadlogger    .setLevel(LogLevel::normal);
  avthreadlogger      .setLevel(LogLevel::normal);
  decoderlogger       .setLevel(LogLevel::normal);
  fifologger          .setLevel(LogLevel::normal);
  opengllogger        .setLevel(LogLevel::normal);
  filelogger          .setLevel(LogLevel::normal);
  filethreadlogger    .setLevel(LogLevel::normal);
  usblogger    .setLevel(LogLevel::normal);
  valkkafslogger    .setLevel(LogLevel::normal);
}

void fatal_log_all() { // only critical / fatal msgs
  framelogger         .setLevel(LogLevel::fatal);
  filterlogger        .setLevel(LogLevel::fatal);
  livelogger          .setLevel(LogLevel::fatal);
  threadlogger        .setLevel(LogLevel::fatal);
  livethreadlogger    .setLevel(LogLevel::fatal);
  avthreadlogger      .setLevel(LogLevel::fatal);
  decoderlogger       .setLevel(LogLevel::fatal);
  fifologger          .setLevel(LogLevel::fatal);
  opengllogger        .setLevel(LogLevel::fatal);
  filelogger          .setLevel(LogLevel::fatal);
  filethreadlogger    .setLevel(LogLevel::fatal);
  usblogger    .setLevel(LogLevel::fatal);
  valkkafslogger    .setLevel(LogLevel::fatal);
}

void quiet_log_all() {
  framelogger         .setLevel(LogLevel::quiet);
  filterlogger        .setLevel(LogLevel::quiet);
  livelogger          .setLevel(LogLevel::quiet);
  threadlogger        .setLevel(LogLevel::quiet);
  livethreadlogger    .setLevel(LogLevel::quiet);
  avthreadlogger      .setLevel(LogLevel::quiet);
  decoderlogger       .setLevel(LogLevel::quiet);
  fifologger          .setLevel(LogLevel::quiet);
  opengllogger        .setLevel(LogLevel::quiet);
  filelogger          .setLevel(LogLevel::quiet);
  filethreadlogger    .setLevel(LogLevel::quiet);
  usblogger    .setLevel(LogLevel::quiet);
  valkkafslogger    .setLevel(LogLevel::quiet);
}
