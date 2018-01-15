%module valkka_core
%include <std_string.i>
%include "cpointer.i" // simple pointer types for c(pp).  We use them for pass-by-reference cases
/* Create some functions for working with "int *" */
%pointer_functions(int, intp);

%{ // this is prepended in the wapper-generated c(pp) file
#define SWIG_FILE_WITH_INIT
#include "include/filters.h"
#include "include/queues.h"
#include "include/livethread.h"
#include "include/avthread.h"
#include "include/openglthread.h"
#include "include/sharedmem.h"
#include "include/logging.h"
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// #define PY_ARRAY_UNIQUE_SYMBOL shmem_array_api
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
%}

%init %{
import_array();            // numpy initialization that should be run only once
ffmpeg_av_register_all();  // register all avcodecs, muxers, etc.
%}

// Swig should not try to create a default constructor for the following classes as they're abstract (swig interface file should not have the constructors either):
%nodefaultctor FrameFilter;
%nodefaultctor Thread;

// XID types
%include<X11/X.h>

%typemap(in) (std::size_t) {
  $1=PyLong_AsSize_t($input);
}


%inline %{

PyObject* getNumpyShmem(SharedMemRingBuffer* rb, int i) {
  PyObject* pa;
  npy_intp dims[1];                                                               
  dims[0]=((rb->shmems)[i])->n_bytes;
  pa=PyArray_SimpleNewFromData(1, dims, NPY_UBYTE, (char*)(((rb->shmems)[i])->payload));
  return pa;                                                                       
}

%}

// next, expose what is necessary
 
class FrameFilter { // <pyapi>
public: // <pyapi>
  virtual ~FrameFilter();                                ///< Virtual destructor // <pyapi>
protected: // <pyapi>
}; // <pyapi>
 
class DummyFrameFilter : public FrameFilter { // <pyapi>
public: // <pyapi>
  DummyFrameFilter(const char* name, bool verbose=true, FrameFilter* next=NULL); ///< @copydoc FrameFilter::FrameFilter // <pyapi>
}; // <pyapi>
 
class InfoFrameFilter : public FrameFilter { // <pyapi>
public: // <pyapi>
  InfoFrameFilter(const char* name, FrameFilter* next=NULL); ///< @copydoc FrameFilter::FrameFilter // <pyapi>
}; // <pyapi>
 
class BriefInfoFrameFilter : public FrameFilter { // <pyapi>
public: // <pyapi>
  BriefInfoFrameFilter(const char* name, FrameFilter* next=NULL); ///< @copydoc FrameFilter::FrameFilter // <pyapi>
}; // <pyapi>
 
class ForkFrameFilter : public FrameFilter { // <pyapi>
public: // <pyapi>
  ForkFrameFilter(const char* name, FrameFilter* next=NULL, FrameFilter* next2=NULL); // <pyapi>
}; // <pyapi>
 
class SlotFrameFilter : public FrameFilter { // <pyapi>
public: // <pyapi>
  SlotFrameFilter(const char* name, SlotNumber n_slot, FrameFilter* next=NULL); ///< @copydoc FrameFilter::FrameFilter // <pyapi>
}; // <pyapi>
 
class DumpFrameFilter : public FrameFilter { // <pyapi>
public: // <pyapi>
  DumpFrameFilter(const char* name, FrameFilter* next=NULL); // <pyapi>
}; // <pyapi>
 
class TimestampFrameFilter : public FrameFilter { // <pyapi>
public: // <pyapi>
  TimestampFrameFilter(const char* name, FrameFilter* next=NULL, long int msdiff_max=1000); ///< @copydoc FrameFilter::FrameFilter // <pyapi>
}; // <pyapi>
 
class RepeatH264ParsFrameFilter : public FrameFilter { // <pyapi>
public: // <pyapi>
  RepeatH264ParsFrameFilter(const char* name, FrameFilter* next=NULL); ///< @copydoc FrameFilter::FrameFilter // <pyapi>
}; // <pyapi>
 
class GateFrameFilter : public FrameFilter { // <pyapi>
public: // <pyapi>
  GateFrameFilter(const char* name, FrameFilter* next=NULL); // <pyapi>
public: // <pyapi>
  void set(); // <pyapi>
  void unSet(); // <pyapi>
}; // <pyapi>
 
class SetSlotFrameFilter : public FrameFilter { // <pyapi>
public: // <pyapi>
  SetSlotFrameFilter(const char* name, FrameFilter* next=NULL); // <pyapi>
public: // <pyapi>
  void setSlot(SlotNumber n=0); // <pyapi>
}; // <pyapi>
 
class TimeIntervalFrameFilter : public FrameFilter { // <pyapi>
public: // <pyapi>
  TimeIntervalFrameFilter(const char* name, long int mstimedelta, FrameFilter* next=NULL); // <pyapi>
}; // <pyapi>
 
class SwScaleFrameFilter : public FrameFilter { // <pyapi>
public: // <pyapi>
  SwScaleFrameFilter(const char* name, int target_width, int target_height, FrameFilter* next=NULL); ///< Default constructor // <pyapi>
  ~SwScaleFrameFilter(); ///< Default destructor // <pyapi>
}; // <pyapi>
 
class FrameFifo { // <pyapi>
public: // <pyapi>
  FrameFifo(const char* name, unsigned short int n_stack); // <pyapi>
  virtual ~FrameFifo(); // <pyapi>
}; // <pyapi>
 
class FifoFrameFilter : public FrameFilter { // <pyapi>
public: // <pyapi>
  FifoFrameFilter(const char* name, FrameFifo& framefifo); ///< Default constructor // <pyapi>
}; // <pyapi>
 
class BlockingFifoFrameFilter : public FrameFilter { // <pyapi>
public: // <pyapi>
  BlockingFifoFrameFilter(const char* name, FrameFifo& framefifo); ///< Default constructor // <pyapi>
}; // <pyapi>
 
class Thread { // <pyapi>
public: // <pyapi>
  virtual ~Thread(); // <pyapi>
public: // *** API *** // <pyapi>
  void startCall(); // <pyapi>
  void stopCall(); // <pyapi>
}; // <pyapi>
 
enum class LiveConnectionType { // <pyapi>
  none,                         // <pyapi>
  rtsp,                         // <pyapi>
  sdp                           // <pyapi>
};                              // <pyapi>
 
struct LiveConnectionContext { // <pyapi>
  LiveConnectionType connection_type; ///< Identifies the connection type                    // <pyapi>
  std::string        address;         ///< Stream address                                    // <pyapi>
  SlotNumber         slot;            ///< A unique stream slot that identifies this stream  // <pyapi>
  FrameFilter*       framefilter;     ///< The frames are feeded into this FrameFilter       // <pyapi>
  // LiveConnectionContext() : connection_type(ConnectionType::none), address(""), slot(0), framefilter(NULL) {} // Initialization to a default value : does not compile ..! // <pyapi>
};                             // <pyapi>
 
class LiveThread : public Thread { // <pyapi>
public:                                                // <pyapi>
  LiveThread(const char* name, int core_id=-1);        // <pyapi>
  ~LiveThread();                                       // <pyapi>
public: // *** C & Python API *** .. these routines go through the convar/mutex locking                                                // <pyapi>
  void registerStreamCall   (LiveConnectionContext &connection_ctx); ///< API method: registers a stream                                // <pyapi> 
  void deregisterStreamCall (LiveConnectionContext &connection_ctx); ///< API method: de-registers a stream                             // <pyapi>
  void playStreamCall       (LiveConnectionContext &connection_ctx); ///< API method: starts playing the stream and feeding frames      // <pyapi>
  void stopStreamCall       (LiveConnectionContext &connection_ctx); ///< API method: stops playing the stream and feeding frames       // <pyapi>
  void stopCall();                                                  ///< API method: stops the LiveThread                              // <pyapi>
}; // <pyapi>
 
class AVThread : public Thread { // <pyapi>
public: // <pyapi>
  AVThread(const char* name, FrameFifo& infifo, FrameFilter& outfilter, int core_id=-1); // <pyapi>
  ~AVThread(); ///< Default destructor // <pyapi>
public: // API <pyapi>
  void decodingOnCall();   ///< API method: enable decoding        // <pyapi>
  void decodingOffCall();  ///< API method: pause decoding         // <pyapi>
  void stopCall();         ///< API method: terminates the thread  // <pyapi>
}; // <pyapi>
 
class OpenGLFrameFifo : public FrameFifo { // <pyapi>
public: // <pyapi>
  OpenGLFrameFifo(unsigned short n_stack_720p, unsigned short n_stack_1080p, unsigned short n_stack_1440p, unsigned short n_stack_4K, unsigned short n_stack_audio); // <pyapi>
  ~OpenGLFrameFifo(); // <pyapi>
}; // <pyapi>
 
class OpenGLThread : public Thread { // <pyapi>
public: // <pyapi>
  OpenGLThread(const char* name, unsigned short n720p=0, unsigned short n1080p=0, unsigned short n1440p=0, unsigned short n4K=0, unsigned short naudio=0, unsigned msbuftime=100, int core_id=-1); // <pyapi>
  virtual ~OpenGLThread(); ///< Virtual destructor // <pyapi>
public: // API // <pyapi>
  OpenGLFrameFifo&    getFifo(); ///< Retrieve the communication fifo  // <pyapi>
public: // for testing // <pyapi>
  Window createWindow();   ///< Creates a new X window (for debugging/testing only) // <pyapi>
  void makeCurrent(const Window window_id);  ///< Set current X windox // <pyapi>
public: // API // <pyapi>
  void stopCall();                                             // <pyapi>
  void infoCall();                                             // <pyapi>
  bool newRenderGroupCall  (Window window_id);                 // <pyapi>
  bool delRenderGroupCall  (Window window_id);                 // <pyapi>
  int newRenderContextCall (SlotNumber slot, Window window_id, unsigned int z);  // <pyapi>
  bool delRenderContextCall(int id);                           // <pyapi>
}; // <pyapi>
static const int VERSION_MAJOR = 0; // <pyapi>
static const int VERSION_MINOR = 2; // <pyapi>
static const int VERSION_PATCH = 0; // <pyapi>
typedef unsigned short SlotNumber;   // <pyapi>
 
class SharedMemRingBuffer { // <pyapi>
public: // <pyapi>
  SharedMemRingBuffer(const char* name, int n_cells, std::size_t n_bytes, int mstimeout=0, bool is_server=false); // <pyapi>
  ~SharedMemRingBuffer(); // <pyapi>
public: // <pyapi>
  int   getValue();       ///< Returns the current index (next to be read) of the shmem buffer // <pyapi>
  bool  getClientState(); ///< Are the shmem segments available for client? // <pyapi>
public: // client side routines - call only from the client side // <pyapi>
  bool clientPull(int &index_out, int &size_out); // <pyapi>
}; // <pyapi>
 
class SharedMemFrameFilter : public FrameFilter { // <pyapi> 
public: // <pyapi>
  SharedMemFrameFilter(const char* name, int n_cells, std::size_t n_bytes, int mstimeout=0); // <pyapi>
}; // <pyapi>
void ffmpeg_av_register_all(); // <pyapi>
void ffmpeg_av_log_set_level(unsigned int level); // <pyapi>
extern void crazy_log_all();   // <pyapi>
extern void debug_log_all();   // <pyapi>
extern void normal_log_all();  // <pyapi>
extern void fatal_log_all();   // <pyapi>
