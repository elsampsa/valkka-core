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



V4LDevice::V4LDevice(const char *dev) : dev(dev), fd(-1), status(v4l_status::none) {
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
    




