/*
 * rawrite_test.cpp : Test O_DIRECT writing class
 * 
 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
 * 
 * Authors: Petri Eranko <petri.eranko@dasys.fi>
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
 *  @file    rawrite_test.cpp
 *  @author  Petri Eranko
 *  @date    2019
 *  @version 1.0.2 
 *  
 *  @brief 
 *
 */ 

#include "logging.h"
#include "avdep.h"
#include "rawrite.h"
#include "test_import.h"


using namespace std::chrono_literals;
using std::this_thread::sleep_for;


void test_1() {
    const char* name = "@TEST: rawrite: test 1: ";
    std::cout << name <<"** @@Write some **" << std::endl;
    int i;
    char dump[5000];
    for(i=0; i<5000; i++) {
        dump[i] = 5;
    }
    
    RaWriter writer = RaWriter("kokkelis.dat");
    
    writer.dump(&dump[0], std::size_t(10));
    writer.dump(&dump[0], std::size_t(5000));
}


void test_2() {
    const char* name = "@TEST: rawrite: test 2: ";
    std::cout << name <<"** @@Write and read some **" << std::endl;
  
    int i;
    char dump[5000];
    for(i=0; i<5000; i++) {
        dump[i] = 5;
    }
    char get[5000];
    for(i=0; i<5000; i++) {
        get[i] = 0;
    }
    
    RaWriter writer = RaWriter("kokkelis.dat");
    
    std::fstream filestream = std::fstream("kokkelis.dat", std::fstream::binary | std::fstream::in);
    filestream.read(&get[0], 10);
    std::cout << "read: " << int(get[0]) << ", " << int(get[1]) << ", " << int(get[2]) << ", " << int(get[3]) << std::endl;
    filestream.close();
    
    writer.dump(&dump[0], std::size_t(10));
    writer.dump(&dump[0], std::size_t(5000));
    writer.close_();
    
    std::fstream filestream2 = std::fstream("kokkelis.dat", std::fstream::binary | std::fstream::in);
    filestream2.read(&get[0], 10);
    std::cout << "read: " << int(get[0]) << ", " << int(get[1]) << ", " << int(get[2]) << ", " << int(get[3]) << std::endl;
    filestream2.close();
}


void test_3() {
    const char* name = "@TEST: rawrite: test 3: ";
    std::cout << name <<"** @@Write'n'read to /dev/sda1 **" << std::endl;
  
    const char* devname = "/dev/sda1";
    
    int i;
    char dump[5000];
    for(i=0; i<5000; i++) {
        dump[i] = 6;
    }
    
    // RaWriter writer = RaWriter(devname, true);
    RaWriter writer = RaWriter(devname, false); // with O_DIRECT
    
    writer.dump(&dump[0], std::size_t(10));
    writer.dump(&dump[0], std::size_t(5000));
}


void test_4() {
    const char* name = "@TEST: rawrite: test 4: ";
    std::cout << name <<"** @@Write'n'read to /dev/sda **" << std::endl;
  
    const char* devname = "/dev/sda";
    
    int i;
    char dump[5000];
    for(i=0; i<5000; i++) {
        dump[i] = 6;
    }
    char get[5000];
    for(i=0; i<5000; i++) {
        get[i] = 0;
    }
    
    // RaWriter writer = RaWriter(devname, true);
    RaWriter writer = RaWriter(devname, false);
    
    std::fstream filestream = std::fstream(devname, std::fstream::binary | std::fstream::in);
    filestream.read(&get[0], 10);
    std::cout << "read: " << int(get[0]) << ", " << int(get[1]) << ", " << int(get[2]) << ", " << int(get[3]) << std::endl;
    filestream.close();
    
    //writer.dump(&dump[0], std::size_t(10));
    //writer.dump(&dump[0], std::size_t(5000));
    
    writer.dump(&dump[0], std::size_t(4096));
    
    writer.close_();
    
    std::fstream filestream2 = std::fstream(devname, std::fstream::binary | std::fstream::in);
    filestream2.read(&get[0], 10);
    std::cout << "read: " << int(get[0]) << ", " << int(get[1]) << ", " << int(get[2]) << ", " << int(get[3]) << std::endl;
    filestream2.close();
}


void test_5() {
    const char* name = "@TEST: rawrite: test 5: ";
    std::cout << name <<"** @@Raw-write'n'read to /dev/sda  **" << std::endl;
  
    const char* devname = "/dev/sda";
    
    int i;
    char dump[5000];
    for(i=0; i<5000; i++) {
        dump[i] = 6;
    }
    
    char get[5000];
    for(i=0; i<5000; i++) {
        get[i] = 0;
    }

    RaWriter  writer = RaWriter(devname, false);
    RawReader reader = RawReader(devname, false);
    
    writer.dump(&dump[0], std::size_t(4096));
    reader.get(&get[0], std::size_t(4096));
    
    std::cout << "read: " << int(get[0]) << ", " << int(get[1]) << ", " << int(get[2]) << ", " << int(get[3]) << std::endl;
    
    writer.close_();
    reader.close_();
}



void test_6() {
    const char* name = "@TEST: rawrite: test 6: ";
    std::cout << name <<"** @@More complicated raw-write'n'read'n'seek to/in /dev/sda  **" << std::endl;
  
    const char* devname = "/dev/sda";
    
    int i;
    char dump[15000];
    for(i=0; i<15000; i++) {
        dump[i] = 1;
    }
    
    char get[15000];
    for(i=0; i<15000; i++) {
        get[i] = 0;
    }

    RaWriter  writer = RaWriter(devname, false);
    RawReader reader = RawReader(devname, false);
    
    i=14;   std::cout << "dump " << i << " bytes\n"; writer.dump(&dump[0], std::size_t(i));
    i=500;  std::cout << "dump " << i << " bytes\n"; writer.dump(&dump[0], std::size_t(i));
    i=4000; std::cout << "dump " << i << " bytes\n"; writer.dump(&dump[0], std::size_t(i));
    i=14;   std::cout << "dump " << i << " bytes\n"; writer.dump(&dump[0], std::size_t(i));
    writer.finish();
    
    std::cout << std::endl;
    
    i=14;   std::cout << "get  " << i << " bytes\n"; reader.get(&get[0], std::size_t(i));
    std::cout << "read: " << int(get[0]) << ", " << int(get[1]) << ", " << int(get[2]) << ", " << int(get[3]) << std::endl;
    i=500;  std::cout << "get  " << i << " bytes\n"; reader.get(&get[0], std::size_t(i));
    std::cout << "read: " << int(get[0]) << ", " << int(get[1]) << ", " << int(get[2]) << ", " << int(get[3]) << std::endl;
    i=4000; std::cout << "get  " << i << " bytes\n"; reader.get(&get[0], std::size_t(i));
    std::cout << "read: " << int(get[0]) << ", " << int(get[1]) << ", " << int(get[2]) << ", " << int(get[3]) << std::endl;
    i=14;   std::cout << "get  " << i << " bytes\n"; reader.get(&get[0], std::size_t(i));
    std::cout << "read: " << int(get[0]) << ", " << int(get[1]) << ", " << int(get[2]) << ", " << int(get[3]) << std::endl;
    
    writer.close_();
    reader.close_();
    
    // TODO: migrate ValkkaFS to RaWriter and RawReader
}



int main(int argc, char** argcv) {
  if (argc<2) {
    std::cout << argcv[0] << " needs an integer argument.  Second interger argument (optional) is verbosity" << std::endl;
  }
  else {
    
    if  (argc>2) { // choose verbosity
      switch (atoi(argcv[2])) {
        case(0): // shut up
          ffmpeg_av_log_set_level(0);
          fatal_log_all();
          break;
        case(1): // normal
          break;
        case(2): // more verbose
          ffmpeg_av_log_set_level(100);
          debug_log_all();
          break;
        case(3): // extremely verbose
          ffmpeg_av_log_set_level(100);
          crazy_log_all();
          break;
        default:
          std::cout << "Unknown verbosity level "<< atoi(argcv[2]) <<std::endl;
          exit(1);
          break;
      }
    }
    
    switch (atoi(argcv[1])) { // choose test
      case(1):
        test_1();
        break;
      case(2):
        test_2();
        break;
      case(3):
        test_3();
        break;
      case(4):
        test_4();
        break;
      case(5):
        test_5();
        break;
      case(6):
        test_6();
        break;
      default:
        std::cout << "No such test "<<argcv[1]<<" for "<<argcv[0]<<std::endl;
    }
  }
} 

/*  Some useful code:


if (!stream_1) {
  std::cout << name <<"ERROR: missing test stream 1: set environment variable VALKKA_TEST_RTSP_1"<< std::endl;
  exit(2);
}
std::cout << name <<"** test rtsp stream 1: "<< stream_1 << std::endl;
  
if (!stream_2) {
  std::cout << name <<"ERROR: missing test stream 2: set environment variable VALKKA_TEST_RTSP_2"<< std::endl;
  exit(2);
}
std::cout << name <<"** test rtsp stream 2: "<< stream_2 << std::endl;
    
if (!stream_sdp) {
  std::cout << name <<"ERROR: missing test sdp file: set environment variable VALKKA_TEST_SDP"<< std::endl;
  exit(2);
}
std::cout << name <<"** test sdp file: "<< stream_sdp << std::endl;

  
*/


