#ifndef LOGGING_HEADER_GUARD 
#define LOGGING_HEADER_GUARD

/*
 * logging.h : Logging utilities
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
 *  @file    logging.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.6.1 
 *  
 *  @brief Logging utilities
 *
 */ 

#include <stdlib.h>
#include <iostream>


namespace LogLevel { ///< Verbosity levels
    const static int quiet =-3;                                                                                 
    const static int fatal =-2; ///< messages with this log level are printed always                                            
    const static int silent=-1; ///< messages with this log level are printed, even if the user requested a silent terminal     
    const static int normal=0;  ///< the default                                                                                
    const static int debug =2;  ///< report frame drops and other anomalies                                                     
    const static int crazy =3;  ///< print all you can                                                                          
}; //                                                                                                                         


/** Logging facility
 * 
 * logger.log(LogLevel::crazy) << a << b << c << etc. gets evaluated as:
 * 
 * ( ( (logger.log(LogLevel::crazy)) << a) << b ) << c )
 * 
 * so first, logger.log(LogLevel::crazy) returns an object that consumers a, which returns an object that consumes b, etc.
 * 
 * For extremely frequent logging, just use std::cout encapsulated inside preprocessor statements
 * 
 */
struct Logger
{
  // Logger(int log_level=LogLevel::normal) : log_level(log_level), current_level(log_level), special_level(0), current_special_level(-1) {};
  Logger(int log_level=LogLevel::normal);

  /*
  template <typename T> // T refers to *any* type
  Logger& operator<<(const T& x);
  */
  
  template <typename T> // T refers to *any* type
  Logger& operator<<(const T& x)
  {
    // std::cout << "current_level, log_level=" << current_level << " " << log_level << std::endl;
    // if (current_level<=log_level or current_special_level==special_level) {
    if (current_level<=log_level) { // current_level is defined at each print .. so, print all messages that are <= the target level
      std::cout << x;
      // std::cout << x << " (c:"<<current_level<<"/ l:"<<log_level<<") "; // debugging the debugging.. eh
    }
    return *this;
  }
  
  int log_level;
  int current_level;
  // int special_level;
  // int current_special_level;
  
  void setLevel(int level);
  
  Logger& log(int level); 
  
  /*
  void setSpecialLevel(int level) { // evokes logs tagged with a special marker
    special_level=level; 
  }
  
  Logger& slog(int level) {
    current_special_level=level;
    return *this;
  }
  */
  
  /* // here could write exceptions to any type ..
  Logger& operator<<(const int& x)
  {
    std::cout << " inteteger: ";
    std::cout << x;
    return *this;
  }
  */
  
  /*
  Logger& operator<<(const LogLevel& x) // nopes.. not like this
  {
    // std::cout << " stopping log ";
    current_level=x;
    return *this;
  }
  */
  
  // function that takes a custom stream, and returns it
  typedef Logger& (*LoggerManipulator)(Logger&); // so.. we create a function type here, because of the brackets () .. a function that eats Logger

  // take in a function with the custom signature
  Logger& operator<<(LoggerManipulator manip); // .. when we encounter that function that eats Logger
  
  // define the custom endl for this stream.
  // note how it matches the `LoggerManipulator`
  // function signature
  static Logger& endl(Logger& stream);
  
  // this is the type of std::cout
  typedef std::basic_ostream<char, std::char_traits<char> > CoutType;

  // this is the function signature of std::endl
  typedef CoutType& (*StandardEndLine)(CoutType&);

  // define an operator<< to take in std::endl
  Logger& operator<<(StandardEndLine manip);
  
};


/**
 * Singleton:
 * https://en.wikipedia.org/wiki/Singleton_pattern
 * https://stackoverflow.com/questions/297822/global-instance-of-a-class-in-c
 * 
 */

extern Logger framelogger;
extern Logger filterlogger;
extern Logger livelogger;
extern Logger threadlogger;
extern Logger livethreadlogger;
extern Logger avthreadlogger;
extern Logger decoderlogger;
extern Logger fifologger;
extern Logger opengllogger;
extern Logger filelogger;
extern Logger filethreadlogger;
extern Logger usblogger;
extern Logger valkkafslogger;

extern void setLogLevel_framelogger(int level);          // <pyapi>
extern void setLogLevel_filterlogger(int level);         // <pyapi>
extern void setLogLevel_livelogger(int level);           // <pyapi>
extern void setLogLevel_threadlogger(int level);         // <pyapi>
extern void setLogLevel_livethreadlogger(int level);     // <pyapi>
extern void setLogLevel_avthreadlogger(int level);       // <pyapi>
extern void setLogLevel_decoderlogger(int level);        // <pyapi>
extern void setLogLevel_fifologger(int level);           // <pyapi>
extern void setLogLevel_opengllogger(int level);         // <pyapi>
extern void setLogLevel_filelogger(int level);           // <pyapi>
extern void setLogLevel_filethreadlogger(int level);     // <pyapi>
extern void setLogLevel_usblogger(int level);            // <pyapi>
extern void setLogLevel_valkkafslogger(int level);       // <pyapi>

extern void crazy_log_all();   // <pyapi>
extern void debug_log_all();   // <pyapi>
extern void normal_log_all();  // <pyapi>
extern void fatal_log_all();   // <pyapi>
extern void quiet_log_all();   // <pyapi>

#endif
