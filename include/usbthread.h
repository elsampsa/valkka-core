#ifndef usbthread_HEADER_GUARD
#define usbthread_HEADER_GUARD
/*
 * usbthread.h : USB Camera control and streaming
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
 *  @file    usbthread.h
 *  @author  Sampsa Riikonen
 *  @date    2018
 *  @version 0.1
 *  
 *  @brief   USB Camera control and streaming
 */ 


#include "common.h"
#include "framefilter.h"
#include "thread.h"

using namespace std::chrono_literals;
using std::this_thread::sleep_for;

#define CLEAR(x) memset(&(x), 0, sizeof(x))

/* TODO

usbthread has a list of devices ..
each device has :
    - a file handle that can be used with select
    - out_frame that can be written to an output framefilter .. should be valid after select has been triggered
    - can be opened and closed
    
    - at thread backend, when incoming request is executed internally, for error messages, could use the python callback

    - could have a std::map<int, Device> (maps fd to device)
    
    
base class: Device
derived class: v4LDevice

*/

class USBDevice {
    
public:    
    USBDevice(FrameFilter *framefilter);
    ~USBDevice();
    
protected:
    int fd;
    FrameFilter *framefilter;   ///< Output FrameFilter
    SetupFrame setupframe;     ///< This frame is used to send subsession information
    BasicFrame basicframe;     ///< Data is being copied into this frame
    
public: // getters
    int getFd() {return fd;}
    
public:
    void setupFrame(SetupFrame frame);  ///< sets the setupframe and writes it to the framefilter
    virtual void open_();
    virtual void close_();
    virtual void pull();            ///< Populates basicframe, sends it through the filter
    virtual void play();
    virtual void stop();
};



// at the python level, check this directory: /sys/class/video4linux
// .. that directory has video0/ video1/   .. with file "name", etc.
// .. /dev/video0 has the actualy device for reading

int xioctl(int fh, int request, void *arg);

enum v4l_status {
    none,           ///< undefined (initial value)
    not_found,      ///< file not found
    not_device,     ///< not device file
    not_read,       ///< could not read device
    not_v4l2,       ///< not v4l2 device
    not_video_cap,  ///< not video capture devices
    not_io,         ///< does not support file io (not used anyway)
    not_stream,     ///< does not support streaming
    not_ptr,        ///< does not support user pointers
    ok_open,        ///< streaming device ok
    not_format,     ///< could not achieve the requested format
    ok_format       ///< everything's ok!
};


class V4LDevice : public USBDevice {
    
public:
    
    /** Default constructor
     * 
     * Open the device, query its capabilities
     * 
     * - Sets opening state
     * 
     * 
     */
    
    V4LDevice(std::string dev, FrameFilter *framefilter);
    ~V4LDevice();
    
protected:
    std::string     dev;     ///< Name of the device
    v4l_status      status;  ///< State of the device
    int             fd;      ///< File number
    v4l2_capability cap;
    v4l2_format     fmt;
    
    
public: // getters
    const v4l_status        getStatus() {return this->status;} 
    const v4l2_capability   getCapability() {return this->cap;}
    
    
public:
    void request(int width, int height, int pix_fmt);
    void initStreaming();
    
};


struct USBCameraConnectionContext { // <pyapi>
    std::string device;             // <pyapi>
    /** A unique stream slot that identifies this stream */
    SlotNumber         slot;        // <pyapi>
    /** Frames are feeded into this FrameFilter */
    FrameFilter*       framefilter; // <pyapi>
    // format, fps, etc.            // <pyapi>
};                                  // <pyapi>


/** Signals used by USBDeviceThread
 */
enum class USBDeviceSignal {
  none,
  exit,
  register_camera_stream,
  deregister_camera_stream,
  play_camera_stream,
  stop_camera_stream
};

  
/** Redefinition of characteristic signal contexts (info that goes with the signal)
*/
struct USBDeviceSignalContext {
  USBDeviceSignal            signal;
  USBCameraConnectionContext camera_connection_ctx;
};


class USBDeviceThread : public Thread {                                               // <pyapi>

public:                                                                               // <pyapi>
    USBDeviceThread(const char *name);                                                // <pyapi>
    ~USBDeviceThread();                                                               // <pyapi>

// protected: // no frame input into this thread
    // FrameFifo               infifo;           ///< Incoming frames are read from here
    // FifoFrameFilter         infilter;         ///< Write incoming frames here
    // BlockingFifoFrameFilter infilter_block;   ///< Incoming frames can also be written here.  If stack runs out of frames, writing will block

protected: // frame output from this thread
    std::map<SlotNumber, USBDevice*>   slots_;          ///< Devices are organized in slots
    
protected: // Thread member redefinitions
    std::deque<USBDeviceSignalContext> signal_fifo;   ///< Redefinition of signal fifo.
  
public: // redefined virtual functions
    void run();
    void preRun();
    void postRun();
    void sendSignal(USBDeviceSignalContext signal_ctx);    ///< Insert a signal into the signal_fifo
      
protected:
    void handleSignal(USBDeviceSignalContext &signal_ctx); ///< Handle an individual signal.  Signal can originate from the frame fifo or from the signal_fifo deque
    void handleSignals();                                  ///< Call USBDeviceThread::handleSignal for every signal in the signal_fifo
    
protected:
    void registerCameraStream   (USBCameraConnectionContext &ctx);    
    void deRegisterCameraStream (USBCameraConnectionContext &ctx);
    void playCameraStream       (USBCameraConnectionContext &ctx);
    void stopCameraStream       (USBCameraConnectionContext &ctx);
    
  
// public API section
public:                                                                 // <pyapi>
    void registerCameraStreamCall   (USBCameraConnectionContext ctx);   // <pyapi>
    void deRegisterCameraStreamCall (USBCameraConnectionContext ctx);   // <pyapi>
    void playCameraStreamCall       (USBCameraConnectionContext ctx);   // <pyapi>
    void stopCameraStreamCall       (USBCameraConnectionContext ctx);   // <pyapi>

    void requestStopCall();                                             // <pyapi>
};                                                                      // <pyapi>







#endif
