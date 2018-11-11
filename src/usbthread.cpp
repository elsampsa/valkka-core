/*
 * usbthread.cpp : USB Camera control and streaming
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
 *  @file    usbthread.cpp
 *  @author  Sampsa Riikonen
 *  @date    2018
 *  @version 0.1
 *  
 *  @brief   USB Camera control and streaming
 */ 

#include "usbthread.h"


int xioctl(int fh, int request, void *arg)
{
    int r;

    do {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);

    return r;
}


USBDevice::USBDevice(FrameFilter *framefilter) : fd(-1), framefilter(framefilter) {
}

USBDevice::~USBDevice() {
}
    
void USBDevice::setupFrame(SetupFrame frame) {
    setupframe=frame;
    framefilter->run(&setupframe);
}    

void USBDevice::open_() {
}

void USBDevice::close_() {
}

void USBDevice::pull() {
    // populate basicframe
    framefilter->run(&basicframe);
}

void USBDevice::play() {
}

void USBDevice::stop() {
}



V4LDevice::V4LDevice(std::string dev, FrameFilter *framefilter) : USBDevice(framefilter),  dev(dev), fd(-1), status(v4l_status::none) {
    struct stat st;
    int min;

    if (stat(this->dev.c_str(), &st) == -1) {
            fprintf(stderr, "Cannot identify '%s': %d, %s\n",
                        this->dev.c_str(), errno, strerror(errno));
        status = v4l_status::not_found;
        return;
    }
    if (!S_ISCHR(st.st_mode)) {
            fprintf(stderr, "%s is no device", this->dev.c_str());
        status = v4l_status::not_device;
        return;
    }

    fd = open(this->dev.c_str(), O_RDWR /* required */ | O_NONBLOCK, 0);

    if (fd == -1 ) {
        fprintf(stderr, "Cannot open '%s': %d, %s\n",
                    this->dev.c_str(), errno, strerror(errno));
        status = v4l_status::not_read;
        return;
    }
    
    /*
    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;
    unsigned int min;
    */
    
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        /*
        if (EINVAL == errno) {
            fprintf(stderr, "%s is no V4L2 device\n", this->dev.c_str());
        } 
        else {
            errno_exit("VIDIOC_QUERYCAP");
        }
        */
        status = v4l_status::not_v4l2;
        return;
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "%s is no video capture device\n", this->dev.c_str());
        status = v4l_status::not_video_cap;
        return;
    }

    /*
    if (!(cap.capabilities & V4L2_CAP_READWRITE)) {
        fprintf(stderr, "%s does not support read i/o\nn", this->dev.c_str());
        status = v4l_status::not_io;
        return;
    }
    */
    
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "%s does not support streaming i/o\n", this->dev.c_str());
        status = v4l_status::not_stream;
        return;
    }
    
    CLEAR(fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    /*
    if (force_format) {
        fmt.fmt.pix.width       = 640;
        fmt.fmt.pix.height      = 480;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;

        if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
                errno_exit("VIDIOC_S_FMT");

        // Note VIDIOC_S_FMT may change width and height
    } 
    else {
    */
    
    // get default parameters
    if (xioctl(fd, VIDIOC_G_FMT, &fmt) == -1 ) {
        std::cout << "V4LDevice: could not get parameters" << std::endl;
        status = v4l_status::not_stream;
        return;
    }
    
    struct v4l2_requestbuffers req;
    CLEAR(req);
    req.count  = 4;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_USERPTR;

    if (xioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        fprintf(stderr, "%s does not support user pointer i/o\n", this->dev.c_str());
        status = v4l_status::not_ptr;
        return;
    }
    
    status = v4l_status::ok_open;

}


void V4LDevice::request(int width, int height, int pix_fmt) {
    
    // https://linuxtv.org/downloads/v4l-dvb-apis/uapi/v4l/vidioc-g-fmt.html
    // https://linuxtv.org/downloads/v4l-dvb-apis/uapi/v4l/userp.html
    
    /* Buggy driver paranoia */
    /*
    min = fmt.fmt.pix.width * 2;
    if (fmt.fmt.pix.bytesperline < min)
        fmt.fmt.pix.bytesperline = min;
    min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
    if (fmt.fmt.pix.sizeimage < min)
        fmt.fmt.pix.sizeimage = min;
    */
    
    // image size is this:
    // fmt.fmt.pix.sizeimage

    // https://stackoverflow.com/questions/10634611/what-is-the-default-format-of-the-raw-data-in-the-external-uvc-usb-camera-yuv
    //
    // Most webcams seem to give only yuv422 (that sucks)
    //
    // yuv422 is a non-planar format (i.e. bytes are not organized in separate planes), instead:
    // YUYV YUYV YUYV YUYV YUYV YUYV
    // (libav : AV_PIX_FMT_YUYV422)
    //
    // AVBitmapFrame implies planar format
    // BitmapPars has parameters for planar format, but that's ok.  Theyre are the "corresponding YUV420P parameters".
    //
    // New class: AVBitmapFrameNP (i.e. "non-planar")
    //
    // OpenGLFrameFifo::writeCopy(Frame *f) ==> *yuvframe =prepareAVBitmapFrame(static_cast<AVBitmapFrame*>(f));
    // ==> chooses frame from the stack, based on f->bmpars 
    // ==> calls YUVFrame::fromAVBitmapFrame (uploads to GPU)
    // prepareAVBitmapFrame / stack choosing and pulling could be done in another subroutine
    // .. then *yuvframe = prepareAVBitmapFrameNP(f)
    //
    // But that would create other kinds of problems
    // For the moment, just use swscale to convert into YUV420P => normal AVBitmapFrame
    //
    // For the *very* moment, just add support for H264 USB cameras
    //
    
    // TODO: split from here to a separate setup call (that then works or not)
    
    // fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    // fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420; // we'd like to have this
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_H264;
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;
    fmt.fmt.pix.width       = 1280; // works!
    fmt.fmt.pix.height      = 720;

    if (xioctl(fd, VIDIOC_S_FMT, &fmt) == -1) { // set the new parameters
        std::cout << "V4L2Device: could not set parameters" << std::endl;
        return;
    }
    
    xioctl(fd, VIDIOC_G_FMT, &fmt); // ok.. let's query the parameters again
    std::cout << "V4L2Device: width, height = " << fmt.fmt.pix.width << " " << fmt.fmt.pix.height << std::endl;
    std::cout << "V4L2Device: sizeimage     = " << fmt.fmt.pix.sizeimage << std::endl;
    if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_H264) {
        std::cout << "V4L2Device: h264 ok" << std::endl;
    } // TODO: check that all parameters were achieved
    
    
    struct v4l2_requestbuffers req;
    CLEAR(req);
    req.count  = 4;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_USERPTR;

    if (xioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        fprintf(stderr, "%s does not support user pointer i/o\n", this->dev.c_str());
        status = v4l_status::not_ptr;
    }

    
}
    
    
void V4LDevice::initStreaming() {
    
    /*
    unsigned int i;
    enum v4l2_buf_type type;

    
    for (i = 0; i < n_buffers; ++i) {
        struct v4l2_buffer buf;

        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_USERPTR;
        buf.index = i;
        buf.m.userptr = (unsigned long)buffers[i].start;
        buf.length = buffers[i].length;

        if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
            std::cout << "V4LDevice: initStreaming: could not map" << std::endl;
            return;
    }
        type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (-1 == xioctl(fd, VIDIOC_STREAMON, &type))
            std::cout << "V4LDevice: initStreaming: vidioc_streamon" << std::endl;
            return;
        
        */
}
    

V4LDevice::~V4LDevice() {
    if (fd > -1) {
        close(fd);
    }
}
    

USBDeviceThread::USBDeviceThread(const char *name) : Thread(name) {
}
    
USBDeviceThread::~USBDeviceThread() {
}

void USBDeviceThread::run() {
    time_t timer;
    time_t oldtimer;
    long int dt;
    
    struct timeval tv;
    int r;
    int tmpfd;
    int fd;
    
    time(&timer);
    oldtimer=timer;
    loop=true;
    
    fd_set fds;
    
    while(loop) {
        sleep_for(0.1s);
        
        FD_ZERO(&fds);
        fd=-1;
        for(auto it=slots_.begin(); it!=slots_.end(); ++it) {
            tmpfd = (it->second)->getFd();
            
            fd=std::max(fd, tmpfd);
            FD_SET(tmpfd, &fds); // add a file descriptor to a set
        }
        
        // TODO: proper timeout
        /* Timeout. */
        tv.tv_sec = 2;
        tv.tv_usec = 0;

        r = select(fd + 1, &fds, NULL, NULL, &tv);

        if (-1 == r) {
            if (EINTR == errno)
                continue;
            // errno_exit("select");
        }
        else if (0 == r) {
            // fprintf(stderr, "select timeout\n");
            // exit(EXIT_FAILURE);
        }
        else {
            // TODO: read the frame
            for(auto it=slots_.begin(); it!=slots_.end(); ++it) {
                if FD_ISSET( (it->second)->getFd(), &fds) {
                    (it->second)->pull(); // pull: populate basicframe and send it down the filterchain
                }
            }
        }
            
        time(&timer);
        // old-style ("interrupt") signal handling
        if (difftime(timer,oldtimer)>=Timeout::usbthread) { // time to check the signals..
            handleSignals();
            oldtimer=timer;
        }
    }
}

void USBDeviceThread::preRun() {
}
    
void USBDeviceThread::postRun() {
}


void USBDeviceThread::registerCameraStream(USBCameraConnectionContext &ctx) {
    auto it=slots_.find(ctx.slot);
    if (it==slots_.end()) { // this slot does not exist
        V4LDevice *device = new V4LDevice(ctx.device, ctx.framefilter);
        slots_.insert(std::make_pair(ctx.slot, device));
        device->open_();
    }
    else {
        std::cout << "USBDeviceThread: registerCameraStream: slot " << ctx.slot << " reserved" << std::endl;
    }
}

void USBDeviceThread::deRegisterCameraStream(USBCameraConnectionContext &ctx) {
    auto it=slots_.find(ctx.slot);
    if (it==slots_.end()) { // this slot does not exist
        std::cout << "USBDeviceThread: deRegisterCameraStream: no such slot " << ctx.slot << std::endl;
    }
    else {
        (it->second)->close_();
        slots_.erase(it);
        delete (it->second);
    }
}

void USBDeviceThread::playCameraStream(USBCameraConnectionContext &ctx) {
    auto it=slots_.find(ctx.slot);
    if (it==slots_.end()) { // this slot does not exist
        std::cout << "USBDeviceThread: playCameraStream: no such slot " << ctx.slot << std::endl;
    }
    else {
        (it->second)->play();
    }
}

void USBDeviceThread::stopCameraStream(USBCameraConnectionContext &ctx) {
    auto it=slots_.find(ctx.slot);
    if (it==slots_.end()) { // this slot does not exist
        std::cout << "USBDeviceThread: stopCameraStream: no such slot " << ctx.slot << std::endl;
    }
    else {
        (it->second)->stop();
    }
}


void USBDeviceThread::handleSignal(USBDeviceSignalContext &signal_ctx) {
    switch (signal_ctx.signal) {
        case USBDeviceSignal::exit:
            loop=false;
            break;
            
        case USBDeviceSignal::register_camera_stream:
            this->registerCameraStream(signal_ctx.camera_connection_ctx);
            break;
            
        case USBDeviceSignal::deregister_camera_stream:
            this->deRegisterCameraStream(signal_ctx.camera_connection_ctx);
            break;
            
        case USBDeviceSignal::play_camera_stream:
            this->playCameraStream(signal_ctx.camera_connection_ctx);
            break;
            
        case USBDeviceSignal::stop_camera_stream:
            this->stopCameraStream(signal_ctx.camera_connection_ctx);
            break;
    }
}

void USBDeviceThread::sendSignal(USBDeviceSignalContext signal_ctx) {
    std::unique_lock<std::mutex> lk(this->mutex);
    this->signal_fifo.push_back(signal_ctx);
}

void USBDeviceThread::handleSignals() {
    std::unique_lock<std::mutex> lk(this->mutex);
    // handle pending signals from the signals fifo
    for (auto it = signal_fifo.begin(); it != signal_fifo.end(); ++it) { // it == pointer to the actual object (struct SignalContext)
        handleSignal(*it);
    }
    signal_fifo.clear();
}


void USBDeviceThread::registerCameraStreamCall(USBCameraConnectionContext ctx) {
    USBDeviceSignalContext signal_ctx;
    signal_ctx.signal = USBDeviceSignal::register_camera_stream;
    sendSignal(signal_ctx);
}

void USBDeviceThread::deRegisterCameraStreamCall(USBCameraConnectionContext ctx) {
    USBDeviceSignalContext signal_ctx;
    signal_ctx.signal = USBDeviceSignal::deregister_camera_stream;
    sendSignal(signal_ctx);
}

void USBDeviceThread::playCameraStreamCall(USBCameraConnectionContext ctx) {
    USBDeviceSignalContext signal_ctx;
    signal_ctx.signal = USBDeviceSignal::play_camera_stream;
    sendSignal(signal_ctx);
}

void USBDeviceThread::stopCameraStreamCall(USBCameraConnectionContext ctx) {
    USBDeviceSignalContext signal_ctx;
    signal_ctx.signal = USBDeviceSignal::stop_camera_stream;
    sendSignal(signal_ctx);
}



void USBDeviceThread::requestStopCall() {
    if (!this->has_thread) { return; } // thread never started
    if (stop_requested) { return; }    // can be requested only once
    stop_requested = true;

    // use the old-style "interrupt" way of sending signals
    USBDeviceSignalContext signal_ctx;
    signal_ctx.signal = USBDeviceSignal::exit;
    
    this->sendSignal(signal_ctx);
}


    

