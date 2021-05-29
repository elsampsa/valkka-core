### Older versions

0.17.0
- Timestamps now taken from decoder!  This means that "main" and "high" H264 profiles with B-frames work.  This should also eliminate some "stuttering" effects seen sometimes in live video.
- "Exotic" bitmaps (YUV422 and other) are now transformed to YUV420, so, for example profiles such as "high422" work (however, this is inefficient, so users should prefer YUV420P streams)

0.16.0
- Yet another memleak at the shmem server side fixed
- Some issues with the python shmem server side fixed

0.15.0
- A nasty memory overflow in the shared memory server / client part fixed
- Added eventfd to the shared mem server / client API.  Now several streams can be multiplexed with select at client side
- Sharing streams between python processes only implemented
- Forgot to call sem_unlink for shared mem semaphores, so they started to cumulate at /dev/shm.  That's now fixed

0.14.1
- Minor changes to valkkafs

0.14.0
- Muxing part progressing (but not yet functional)
- python API 2 level updates

0.13.2 Version
- Extracting SPS & PPS packets from RTSP negotiation was disabled..!
- Now it's on, so cameras that don't send them explicitly (like Axis) should work

0.13.1 Version
- Matroska export from ValkkaFS, etc.
- Lightweight OnVif client

0.12.0 Version
- Shared memory frame transport now includes more metadata about the frames: slot, timestamp, etc.  Now it also works with python multithreading.
- Numpy was included to valkkafs in an incorrect way, this might have resulted in mysterious segfaults.  Fixed that.
- At valkka-examples, fixed the multiprocessing/analyzer example (fork first, then spawn threads)

0.11.0 Version
- Bug fixes at the live555 bridge by Petri
- ValkkaFS progressing
- Currently Ubuntu PPA build fails for i386 & armhf.  This has something to do with the ```off_t``` datatype ?

0.10.0 Version
- Nasty segmentation fault in OpenGL part fixed: called glDeleteBuffers instread of glDeleteVertexArrays for a VAO !
- ValkkaFS progressing

0.9.0 Version
- H264 USB Cameras work

0.8.0 Version : "Say Yolo again"
- Added bounding boxes to OpenGLThread API
- Tested Valkka Live with YOLOv3 object detection
- Started sketching USB camera thread and ValkkaFS

0.7.1 Version
- Small fix in the python part valkka.api2 (in managed filterchain)

0.7.0 Version : "One namespace to rule them all"
- Switched to python namespace packaging

0.6.0 Version
- etc

0.5.4 Version
- Python API-level 2 : A managed filterchain that handles streams on-demand between processes

0.5.3 Version
- Woops .. there was no regression but a nasty bug with the use of stl containers
- .. works fine now, after I followed the "rule of four" (ctor, dtor, copy-ctor and copy-assignment)

0.5.2 Version
- Fixed a small bug: smart timestamp correction is again the default
- Weird regression here: segfault with intel gfx drivers with 5+ streams.  Problem with the latest intel driver?

0.5.1 Version : "Videowalls"
- Multi-GPU works

0.5.0 Version : "Halfway to Beta"
- Live rendering very solid

0.4.7 Version
- Can change receiving socket size and Live555 reordering buffer
- etc.

0.4.6 Version
- Background texture when no stream is coming through
- Weird bitmap sizes work
- TestThread class for sending (PyQt) signals from cpp
- etc.

0.4.4 Version
- RTSP server works

0.4.3 Version
- Now reads acc in addition to annex b h264

0.4.0 Version : "The Rewrite"
- A complete rewrite of the library
- (documentation might lag a bit behind..)

0.3.6 Version
- Sending multicast works

0.3.5 Version : "10 x 1080p cameras running for a week"
- Stable!  GPU direct memory access needed rubbing in the right way (..or did it?)
- Lots of fixes ..
- Reading / writing from/to matroska container

0.3.0 Version name : "It was all about the vblank"
- Several full-HD cameras now streaming OK
- Interoperability with python multiprocesses (and from there, with OpenCV)
- For benchmarking, testing and demos see the "valkka-examples" repository

0.2.1 Version
- License change (to APGL)
- Added python level 2 api example
- Miscellaneous fixes

0.2.0 Version name : "Christmas 2017 project"
- Software interpolator filter (yuv => rgb interpolation in the CPU)
- Shared memory bridge for python inter-process communication
- Python level 2 api
- Just committed this one : documentation and packages will be updated soon :)

0.1.0 Version name : "Proof of concept"
- Initial git commit: core system, live streaming to X-window system 
