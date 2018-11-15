struct stat st;
    int min;

    if (stat(camera_ctx.device.c_str(), &st) == -1) {
            fprintf(stderr, "Cannot identify '%s': %d, %s\n",
                        camera_ctx.device.c_str(), errno, strerror(errno));
        status = v4l_status::not_found;
        return;
    }
    if (!S_ISCHR(st.st_mode)) {
            fprintf(stderr, "%s is no device", camera_ctx.device.c_str());
        status = v4l_status::not_device;
        return;
    }

    fd = open(camera_ctx.device.c_str(), O_RDWR /* required */ | O_NONBLOCK, 0);

    if (fd == -1 ) {
        fprintf(stderr, "Cannot open '%s': %d, %s\n",
                    camera_ctx.device.c_str(), errno, strerror(errno));
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
            fprintf(stderr, "%s is no V4L2 device\n", camera_ctx.device.c_str());
        } 
        else {
            fprintf(stderr,"VIDIOC_QUERYCAP");
        }
        */
        status = v4l_status::not_v4l2;
        return;
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "%s is no video capture device\n", camera_ctx.device.c_str());
        status = v4l_status::not_video_cap;
        return;
    }

    /*
    if (!(cap.capabilities & V4L2_CAP_READWRITE)) {
        fprintf(stderr, "%s does not support read i/o\nn", camera_ctx.device.c_str());
        status = v4l_status::not_io;
        return;
    }
    */
    
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "%s does not support streaming i/o\n", camera_ctx.device.c_str());
        status = v4l_status::not_stream;
        return;
    }
    
    struct v4l2_requestbuffers req;
    CLEAR(req);
    req.count  = 4;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_USERPTR;

    if (xioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        fprintf(stderr, "%s does not support user pointer i/o\n", camera_ctx.device.c_str());
        status = v4l_status::not_ptr;
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
                fprintf(stderr,"VIDIOC_S_FMT");

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
    
    status = v4l_status::ok_open;
    
    std::cout << "V4LDevice: device open!" << std::endl;
    
    CLEAR(fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    // fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    // fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420; // we'd like to have this
    /*
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_H264;
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;
    fmt.fmt.pix.width       = 1280; // works!
    fmt.fmt.pix.height      = 720;
    */
    
    //fmt.fmt.pix.width       = 640;
    //fmt.fmt.pix.height      = 480;
    
    fmt.fmt.pix.width       = 1280;
    fmt.fmt.pix.height      = 720;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;


    // this fucker is never happy
    if (xioctl(fd, VIDIOC_S_FMT, &fmt) == -1) { // set the new parameters
        std::cout << "V4L2Device: could not set parameters " << camera_ctx.device << std::endl;
        // return;
    }
    
    
    xioctl(fd, VIDIOC_G_FMT, &fmt); // ok.. let's query the parameters again
    std::cout << "V4L2Device: width, height = " << fmt.fmt.pix.width << " " << fmt.fmt.pix.height << std::endl;
    std::cout << "V4L2Device: sizeimage     = " << fmt.fmt.pix.sizeimage << std::endl;
    if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_H264) {
        std::cout << "V4L2Device: h264 ok" << std::endl;
    } // TODO: check that all parameters were achieved
    
    // struct v4l2_requestbuffers req;
    CLEAR(req);
    req.count  = 4;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_USERPTR;

    if (xioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        fprintf(stderr, "%s does not support user pointer i/o\n", camera_ctx.device.c_str());
        status = v4l_status::not_ptr;
    } 
