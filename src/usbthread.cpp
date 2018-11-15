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


struct buffer {
        void   *start;
        size_t  length;
};



int xioctl(int fh, int request, void *arg)
{
    int r;

    do {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);

    return r;
}


USBDevice::USBDevice(FrameFilter *framefilter) : fd(-1), framefilter(framefilter), playing(false) {
}

USBDevice::~USBDevice() {
}
    
void USBDevice::setupFrame(SetupFrame frame) {
    setupframe=frame;
    framefilter->run(&setupframe);
}    

void USBDevice::open_() {
    this->close_();
}

void USBDevice::close_() {
    this->stop();
}

int USBDevice::pull() {
    // populate basicframe
    framefilter->run(&basicframe);
    return -1;
}

void USBDevice::play() {
    playing=true;
}

void USBDevice::stop() {
    playing=false;
}



V4LDevice::V4LDevice(USBCameraConnectionContext camera_ctx) : USBDevice(camera_ctx.framefilter),  camera_ctx(camera_ctx), status(v4l_status::none) {
    BasicFrame *f;
    int i;
    for(i=0;i<n_ring_buffer;i++) {
        f = new BasicFrame();
        // f->resize(PayloadSizes::DEFAULT_PAYLOAD_SIZE_H264);
        ring_buffer.push_back(f);
    }
}

V4LDevice::~V4LDevice() {
    std::cout << "V4LDevice : dtor" << std::endl;
    if (status >= v4l_status::ok_open) {
        close_();
    }
    for(auto it=ring_buffer.begin(); it!=ring_buffer.end(); it++) {
        delete *it;
    }
}


void V4LDevice::open_() {
    struct stat st;
    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;
    struct v4l2_requestbuffers req;
    // struct v4l2_buffer buf;
    enum v4l2_buf_type type;
    unsigned int min;
    bool force_format = true;
    
    this->close_();
    
    if (stat(camera_ctx.device.c_str(), &st) == -1) {
        std::cout << "Cannot identify " << camera_ctx.device << std::endl;
        //fprintf(stderr, "Cannot identify '%s': %d, %s\n",
        //            camera_ctx.device.c_str(), errno, strerror(errno));
        status = v4l_status::not_found;
        return;
    }
    if (!S_ISCHR(st.st_mode)) {
        std::cout << camera_ctx.device << " is not a device" << std::endl;
        // fprintf(stderr, "%s is no device", camera_ctx.device.c_str());
        status = v4l_status::not_device;
        return;
    }

    fd = open(camera_ctx.device.c_str(), O_RDWR | O_NONBLOCK, 0);

    if (fd == -1 ) {
        std::cout << "Cannot open " << camera_ctx.device << std::endl;
        //fprintf(stderr, "Cannot open '%s': %d, %s\n",
        //            camera_ctx.device.c_str(), errno, strerror(errno));
        status = v4l_status::not_read;
        return;
    }

    status = v4l_status::ok_open;
    
    /*
    const char* dev_name = "/dev/video2";
    fd = open(dev_name, O_RDWR | O_NONBLOCK, 0);
    */
    
    if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap)) {
        if (EINVAL == errno) {
            std::cout << camera_ctx.device << " is no V4l2 device " << std::endl;
            // fprintf(stderr, "is no V4L2 device\\n");
            // exit(EXIT_FAILURE);
            status = v4l_status::not_v4l2;
            return;
        } 
        else {
            // fprintf(stderr,"VIDIOC_QUERYCAP");
            status = v4l_status::not_v4l2;
            return;
        }
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        std::cout << camera_ctx.device << " is not video capture device " << std::endl;
        status = v4l_status::not_video_cap;
        return;
        // fprintf(stderr, "is no video capture device\\n");
        // exit(EXIT_FAILURE);
    }

    
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        std::cout << camera_ctx.device << " does not support streaming " << std::endl;
        status = v4l_status::not_stream;
        return;
        //fprintf(stderr, "does not support streaming i/o\\n");
        //exit(EXIT_FAILURE);
    }
        
    /* Select video input, video standard and tune here. */

    /*
    CLEAR(cropcap);

    cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (0 == xioctl(fd, VIDIOC_CROPCAP, &cropcap)) {
            crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            crop.c = cropcap.defrect; // reset to default

            if (-1 == xioctl(fd, VIDIOC_S_CROP, &crop)) {
                    switch (errno) {
                    case EINVAL:
                            // Cropping not supported
                            break;
                    default:
                            // Errors ignored.
                            break;
                    }
            }
    } else {
            // Errors ignored
    }
    */

    CLEAR(fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    // if (force_format) { // FORCE_FORMAT
    // fmt.fmt.pix.width       = 640;
    // fmt.fmt.pix.height      = 480;
    
    // fmt.fmt.pix.width       = 1280;
    // fmt.fmt.pix.height      = 720;
    
    // fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    // fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;
    
    fmt.fmt.pix.width       = camera_ctx.width;
    fmt.fmt.pix.height      = camera_ctx.height;
    
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_H264;
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;
    

    if (xioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        std::cout << "Could not set format for " << camera_ctx.device << std::endl;
        status = v4l_status::not_format;
        return;
        // fprintf(stderr,"VIDIOC_S_FMT");
    }
    /* Note VIDIOC_S_FMT may change width and height. */
    // } // FORCE_FORMAT
    /*
    else {
        // Preserve original settings as set by v4l2-ctl for example
        if (-1 == xioctl(fd, VIDIOC_G_FMT, &fmt)) {
            fprintf(stderr,"VIDIOC_G_FMT");
        }
    }
    */
    
    std::cout << "V4LDevice : image size : " << fmt.fmt.pix.sizeimage << std::endl;
    
    status = v4l_status::ok_format;

    // int n_buffers = 4;
    
    CLEAR(req);

    // req.count  = n_buffers;
    req.count  = ring_buffer.size();
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_USERPTR;

    if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
        if (EINVAL == errno) {
            std::cout << camera_ctx.device <<" does not support user pointer i/o" << std::endl;
            status = v4l_status::not_ptr;
            return;
            // fprintf(stderr, "%s does not support user pointer i/on", dev_name);
            // exit(EXIT_FAILURE);
        } 
        else {
            std::cout << camera_ctx.device <<" does not support user pointer i/o" << std::endl;
            status = v4l_status::not_ptr;
            return;
            // errno_exit("VIDIOC_REQBUFS");
        }
    }
    
    
    /*
    struct buffer *buffers;
    // int buffer_size = 1024*1024*10; // yes! :)
    int buffer_size = fmt.fmt.pix.sizeimage;
    
    buffers = (buffer*)calloc(4, sizeof(*buffers));

    if (!buffers) {
            fprintf(stderr, "Out of memory\\n");
            exit(EXIT_FAILURE);
    }

    for (n_buffers = 0; n_buffers < 4; ++n_buffers) {
            buffers[n_buffers].length = buffer_size;
            buffers[n_buffers].start = malloc(buffer_size);

            if (!buffers[n_buffers].start) {
                    fprintf(stderr, "Out of memory\\n");
                    exit(EXIT_FAILURE);
            }
    }
    
    
    int i;
    for (i = 0; i < n_buffers; ++i) {
        // struct v4l2_buffer buf;

        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_USERPTR;
        buf.index = i;
        buf.m.userptr = (unsigned long)buffers[i].start;
        std::cout << "ptr>" << buf.m.userptr << std::endl;
        buf.length = buffers[i].length;
        std::cout << "len>" << buf.length << std::endl;
        
        if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
            std::cout << "VIDIOC_QBUF" << std::endl;;
                // errno_exit("VIDIOC_QBUF");
    }
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(fd, VIDIOC_STREAMON, &type))
            // errno_exit("VIDIOC_STREAMON");
            std::cout << "VIDIOC_STREAMON" << std::endl;;
    */
    
    
    ///*
    int cc=0;
    for(auto it=ring_buffer.begin(); it!=ring_buffer.end(); it++) {
        std::cout << "V4LDevice: setting ring_buffer " << cc << std::endl;
        (*it)->reserve(fmt.fmt.pix.sizeimage);
        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_USERPTR;
        buf.index = cc;
        buf.m.userptr = (unsigned long)( (*it)->payload.data() );
        buf.length = ((*it)->payload).capacity();
        // buf.length = fmt.fmt.pix.sizeimage;
        if (-1 == xioctl(fd, VIDIOC_QBUF, &buf)) {
            std::cout << "V4LDevice: initStreaming: could not map" << std::endl;
            status = v4l_status::not_map;
            return;
        }
        cc++;
    }
    
    
    // it would make more sense to use the following semantics:
    // play == VIDIOC_STREAMON
    // stop == VIDIOC_STREAMOFF
    //
    // .. but once VIDIOC_STREAMOFF has been called, my test camera goes nuts if you try to do again VIDIOC_STREAMON
    // .. stupid, buggy drivers
    //
    // enum v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(fd, VIDIOC_STREAMON, &type)) {
        // errno_exit("VIDIOC_STREAMON");
        std::cout << "V4LDevice: open_: VIDIOC_STREAMON" << std::endl;
        status = v4l_status::not_on;
        return;
    }
    
    status = v4l_status::ok;
}

void V4LDevice::close_() {
    this->stop();
    if (status >= v4l_status::ok_open) {
        std::cout << "V4LDevice: close_: closing device" << std::endl;
        close(fd);
    }
    if (status >= v4l_status::ok) {
        enum v4l2_buf_type type;
        type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (-1 == xioctl(fd, VIDIOC_STREAMOFF, &type) ) {
            std::cout << "V4LDevice: close_: VIDIOC_STREAMOFF" << std::endl;
            // exit(2);
        }
    }
    status = v4l_status::none;
}

int V4LDevice::pull() {
    // populate basicframe
    
    CLEAR(buf);

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_USERPTR;

    if (-1 == xioctl(fd, VIDIOC_DQBUF, &buf)) {
            switch (errno) {
            case EAGAIN:
                return 0;

            case EIO:
                /* Could ignore EIO, see spec. */
                /* fall through */

            default:
                // errno_exit("VIDIOC_DQBUF");
                std::cout << "V4LDevice: pull: failed" << std::endl;
                return -1;
            }
    }
    // why v4l2 does not give the index directly?
    int i;
    for (i = 0; i < ring_buffer.size(); ++i) {
        // if (buf.m.userptr == (unsigned long)buffers[i].start && buf.length == buffers[i].length)
        // break;
        if ( buf.m.userptr == (unsigned long)(ring_buffer[i]->payload.data()) ) {
            std::cout << "V4LDevice: pull: got bytes: " << buf.bytesused << std::endl;
            ring_buffer[i]->payload.resize(buf.bytesused);
            std::cout << "V4LDevice: pull: payload  : " << ring_buffer[i]->dumpPayload() << std::endl;
            break;
        }
    }

    std::cout << "V4LDevice: pull: ring buffer index: " << i << std::endl;
    // framefilter->run(&basicframe);
    
    if (-1 == xioctl(fd, VIDIOC_QBUF, &buf)) {
        // errno_exit("VIDIOC_QBUF");
        std::cout << "VIDIOC_QBUF" << std::endl;
        return -1;
    }
        
    return i;
}

void V4LDevice::play() {
    std::cout << "V4LDevice: play" << std::endl;
    this->stop();
    this->open_();
    if (status == v4l_status::ok) {
        playing=true;
    }
    else {
        this->close_();
        playing=false;
    }
}

void V4LDevice::stop() {
    if (playing) {
        std::cout << "V4LDevice: stop: was playing" << std::endl;
        playing=false;
        this->close_();
    }
}


USBDeviceThread::USBDeviceThread(const char *name) : Thread(name) {
}
    
USBDeviceThread::~USBDeviceThread() {
}

void USBDeviceThread::run() {
    long int dt;
    long int mstime, oldmstime;
    
    struct timeval tv;
    int r;
    int tmpfd;
    int fd;

    mstime = getCurrentMsTimestamp();
    oldmstime = mstime;
    
    fd_set fds;
    tv = msToTimeval(Timeout::usbthread);
    loop=true;
    
    while(loop) {
        std::cout << "USBDeviceThread: loop, dt=" << dt << std::endl;
        FD_ZERO(&fds);
        fd=-1;
        for(auto it=slots_.begin(); it!=slots_.end(); ++it) {
            if (it->second->isPlaying()) {
                tmpfd = (it->second)->getFd();
                std::cout << "USBDeviceThread: run: isPlaying: fd=" << tmpfd << std::endl;
                fd=std::max(fd, tmpfd);
                FD_SET(tmpfd, &fds); // add a file descriptor to the set
            }
        }
        
        tv = msToTimeval(Timeout::usbthread); // must be set each time
        
        r = select(fd + 1, &fds, NULL, NULL, &tv);

        if (-1 == r) {
            if (EINTR == errno)
                continue;
            // fprintf(stderr,"select");
        }
        else if (0 == r) {
            // fprintf(stderr, "select timeout\n");
            // exit(EXIT_FAILURE);
        }
        else {
            // TODO: read the frame
            for(auto it=slots_.begin(); it!=slots_.end(); ++it) {
                if FD_ISSET( (it->second)->getFd(), &fds) {
                    std::cout << "USBDeviceThread: run: pulling frame" << std::endl;
                    int num = (it->second)->pull(); // pull: populate basicframe and send it down the filterchain
                    std::cout << "USBDeviceThread: pull returned " << num << std::endl;
                    if (num<0) {
                        std::cout << "USBDeviceThread: run: FATAL: pull failed" << std::endl;
                        (it->second)->close_();
                    }
                }
            }
        }
            
        mstime = getCurrentMsTimestamp();
        dt = mstime-oldmstime;
        // old-style ("interrupt") signal handling
        if (dt>=Timeout::usbthread) { // time to check the signals..
            std::cout << "USBDeviceThread: run: interrupt, dt= " << dt << std::endl;
            handleSignals();
            oldmstime=mstime;
        }
    }
}

void USBDeviceThread::preRun() {
}
    
void USBDeviceThread::postRun() {
    std::cout << "USBDeviceThread: postRun" << std::endl;
    for (auto it=slots_.begin(); it!=slots_.end(); it++) {
        // (it->second)->close_();
        (it->second)->stop();
        delete it->second;
    }    
}


//void USBDeviceThread::registerCameraStream(USBCameraConnectionContext &ctx) {
void USBDeviceThread::playCameraStream(USBCameraConnectionContext &ctx) {
    auto it=slots_.find(ctx.slot);
    if (it==slots_.end()) { // this slot does not exist
        V4LDevice *device = new V4LDevice(ctx);
        slots_.insert(std::make_pair(ctx.slot, device));
        // device->open_();
        device->play();
    }
    else {
        std::cout << "USBDeviceThread: registerCameraStream: slot " << ctx.slot << " reserved" << std::endl;
    }
}

// void USBDeviceThread::deRegisterCameraStream(USBCameraConnectionContext &ctx) {
void USBDeviceThread::stopCameraStream(USBCameraConnectionContext &ctx) {
    auto it=slots_.find(ctx.slot);
    if (it==slots_.end()) { // this slot does not exist
        std::cout << "USBDeviceThread: deRegisterCameraStream: no such slot " << ctx.slot << std::endl;
    }
    else {
        // (it->second)->close_();
        (it->second)->stop();
        slots_.erase(it);
        delete (it->second);
    }
}

/*
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
*/


void USBDeviceThread::handleSignal(USBDeviceSignalContext &signal_ctx) {
    switch (signal_ctx.signal) {
        case USBDeviceSignal::exit:
            loop=false;
            break;
            
        /*
        case USBDeviceSignal::register_camera_stream:
            this->registerCameraStream(signal_ctx.camera_connection_ctx);
            break;
            
        case USBDeviceSignal::deregister_camera_stream:
            this->deRegisterCameraStream(signal_ctx.camera_connection_ctx);
            break;
        */
            
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


/*
void USBDeviceThread::registerCameraStreamCall(USBCameraConnectionContext ctx) {
    USBDeviceSignalContext signal_ctx;
    signal_ctx.signal = USBDeviceSignal::register_camera_stream;
    signal_ctx.camera_connection_ctx = ctx;
    sendSignal(signal_ctx);
}

void USBDeviceThread::deRegisterCameraStreamCall(USBCameraConnectionContext ctx) {
    USBDeviceSignalContext signal_ctx;
    signal_ctx.signal = USBDeviceSignal::deregister_camera_stream;
    signal_ctx.camera_connection_ctx = ctx;
    sendSignal(signal_ctx);
}
*/

void USBDeviceThread::playCameraStreamCall(USBCameraConnectionContext ctx) {
    USBDeviceSignalContext signal_ctx;
    signal_ctx.signal = USBDeviceSignal::play_camera_stream;
    signal_ctx.camera_connection_ctx = ctx;
    sendSignal(signal_ctx);
}

void USBDeviceThread::stopCameraStreamCall(USBCameraConnectionContext ctx) {
    USBDeviceSignalContext signal_ctx;
    signal_ctx.signal = USBDeviceSignal::stop_camera_stream;
    signal_ctx.camera_connection_ctx = ctx;
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


    

