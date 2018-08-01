#ifndef openglthread_HEADER_GUARD
#define openglthread_HEADER_GUARD
/*
 * openglthread.h : The OpenGL thread for presenting frames and related data structures
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
 *  @file    openglthread.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.5.2 
 *  
 *  @brief The OpenGL thread for presenting frames and related data structures
 *
 */ 

#include "tex.h"
#include "openglframefifo.h"
#include "constant.h"
#include "thread.h"
#include "shader.h"
#include "threadsignal.h"


typedef void ( *PFNGLXSWAPINTERVALEXTPROC) (Display *dpy, GLXDrawable drawable, int interval);
// typedef void ( *PFNGLXSWAPINTERVALEXTPROC) (Display *dpy, GLXDrawable drawable, int interval);

/** Each Frame carries information about it's slot number in Frame::slot.  Slot number identifies the origin of the stream.  When the bitmap Frame arrives to OpenGLThread, each stream/slot requires:
 * 
 * - A set of OpenGL Textures (SlotContext::yuvtex)
 * - A reference to the relevant shader program (SlotContext::shader)
 * 
 * This class encapsulates all that.  SlotContext can be in an active (with textures reserved and shaders set) or in an inactive state.
 * 
 * @ingroup openglthread_tag
 */
class SlotContext {
 
public:
  SlotContext(YUVTEX *statictex, YUVShader *shader);  ///< Default constructor
  ~SlotContext(); ///< Default destructor
  
public:  
  YUVTEX*      yuvtex;      ///< This could be a base class for different kinds of textures (now only YUVTEX)
  YUVTEX*      statictex; ///< A static texture to be shown on the screen if no video is received
  YUVShader*   shader;      ///< Base class for the chosen Shader for this slot.  Now always YUVShader
  BitmapPars   bmpars;
  long int     lastmstime;  ///< Last millisecond timestamp when this slot received a frame
    
private:
  uint      ref_count;
  bool      active;     ///< Is activated or not.  Active = has received a setup frame
  bool      is_dead;    ///< Has received frames or not for a while ?
  AVCodecID codec_id;   ///< FFmpeg codec id
  
private:
  long int  prev_mstimestamp; ///< for debugging: check if frames are fed in correct timestamp order
  
public:
  // void activate(BitmapPars bmpars, YUVShader* shader);      ///< Allocate SlotContext::yuvtex (for a frame of certain size) and SlotContext::shader.
  void activate(BitmapPars bmpars);                         ///< Allocate SlotContext::yuvtex (for a frame of certain size)
  void deActivate();                                        ///< Deallocate textures
  void loadYUVFrame(YUVFrame* yuvframe);                    ///< Load bitmap from YUVFrame to SlotContext::yuvtex
  YUVTEX* getTEX();                                         ///< Returns the relevant texture set (static or live)
  bool manageTimer(long int mstime);                        ///< Updates SlotContext::lastmstime.  Returns false if too little time has passed since receiving the last frame
  void checkIfDead(long int mstime);                        ///< Compares SlotContext::lastmstime to mstime and changes the state of SlotContext::is_dead
  bool isPending(long int mstime);                          ///< Compares SlotContext::lastmstime to mstime and returns true of a treshold is met (if last frame was received long time ago)
    
public: // getters
  bool isUsed()   const {return ref_count>0;}
  bool isActive() const {return active;}   ///< Check if active
  bool isDead()   const {return is_dead;}
  
public: // setters
  void inc_ref_count() {ref_count++;}
  void dec_ref_count() {ref_count--;}
};



/** Encapsulates data for rendering a single bitmap: vertex array object (VAO), vertex buffer object (VBO), vertice coordinates, transformation matrix, etc.
 * 
 * @ingroup openglthread_tag
 */
class RenderContext {

public:
  /** Default constructor
   * 
   * @param slot_context A reference to a SlotContext instance
   * @param z            Stacking order (for "video-in-video")
   * 
   */
  RenderContext(SlotContext& slot_context, int id, unsigned int z=0);
  virtual ~RenderContext(); ///< Default virtual destructor
  
public: // Initialized at constructor init list or at constructor
  SlotContext& slot_context;       ///< Knows relevant Shader and YUVTEX
  const unsigned int z;            ///< Stacking order
  // bool active;
  int id;                          ///< A unique id identifying this render context
  
private: // Initialized at RenderContext::init that calls RenderContext::activate
  GLuint        VAO;     ///< id of the vertex array object
  GLuint        VBO;     ///< id of the vertex buffer object
  GLuint        EBO;     ///< id of the element buffer object
  std::array<GLfloat,16> transform; ///< data of the transformation matrix
  std::array<GLfloat,20> vertices;  ///< data of the vertex buffer object
  std::array<GLuint, 6>  indices;   ///< data of the element buffer object
  
private:
  void activate();   ///< Init VAO, VBO, etc.
    
public:
  XWindowAttributes x_window_attr;  ///< X window size, etc.
  
public: // getters
  int getId() const {return id;}        ///< Returns the unique id of this render context
  // bool isActive() {return active;}   ///< Legacy.  Now, if instantiated, is active!
  
public:
  // bool activate();        ///< Activate this RenderContext (now it can render).  Legacy
  // bool activateIf();      ///< Try to activate if not active already. Legacy
  void render(XWindowAttributes x_window_attr);  ///< Does the actual drawing of the bitmap
  void bindTextures();    ///< Associate textures with the shader program.  Uses parameters from Shader reference.
  void bindVars();        ///< Upload other data to the GPU (say, transformation matrix).  Uses parameters from Shader reference.
  void bindVertexArray(); ///< Bind the vertex array and draw
};

// https://stackoverflow.com/questions/4421706/what-are-the-basic-rules-and-idioms-for-operator-overloading
// inline bool operator==(const RenderContext& lhs, const RenderContext& rhs){ return (lhs.getId()==rhs.getId()); }


/** Group of bitmaps that are rendered into the same X-window.  A RenderGroup instance is indentified uniquely by an x-window id.
 * 
 * @ingroup openglthread_tag
 */ 
class RenderGroup {
  
public:
  /** Default constructor
   * 
   * @param display_id        Pointer to x display id
   * @param glc               Reference to GXLContext
   * @param window_id         X window id: used as an id for the render group
   * @param child_id          X window id: this is the actual render target.  Can be the same as window_id.
   * @param doublebuffer_flag Use double buffering or not (default true)
   */
  RenderGroup(Display *display_id, const GLXContext &glc, Window window_id, Window child_id, bool doublebuffer_flag=true);
  ~RenderGroup();        ///< Default destructor
  
public:
  Display* display_id;    ///< X display id
  const GLXContext &glc;  ///< GLX Context
  Window   window_id;     ///< X window id: render group id
  Window   child_id;      ///< X window id: rendering target
  bool     doublebuffer_flag; ///< Double buffer rendering or not?
  std::list<RenderContext> render_contexes; ///< RenderContext instances in ascending z order.  User created rendercontexes are warehoused here
  XWindowAttributes x_window_attr;          ///< X window attributes
  
public: // getters
  Window getWindowId() const {return window_id;}
  const std::list<RenderContext>& getRenderContexes() const {return render_contexes;}
  
public:
  std::list<RenderContext>::iterator getContext(int id); ///< Returns iterator at matched render context id
  bool addContext(RenderContext render_context);         ///< Add RenderContext to RenderGroup::render_contexes (with checking)
  bool delContext(int id);                               ///< Remove RenderContext from RenderGroup::render_contexes (with checking)
  bool isEmpty();                                        ///< Checks if there are any render contexes in the render_contexes list
  /** Render everything in this group (i.e. in this x window)
   * 
   * - Set current x window to RenderGroup::window_id
   * - Set viewport
   * - Clear drawing buffer
   * - Run over RenderGroup::render_contexes and call each one's RenderContext::render() method
   */
  void render();
};


/** GLX extensions for controlling the vertical sync / framerate issue are a mess.  Hence this namespace.
 * 
 */
namespace swap_flavors {
  const static unsigned     none =0;
  const static unsigned     ext  =1;
  const static unsigned     mesa =2;
  const static unsigned     sgi  =3;
}


/** This class does a lot of things:
 * 
 * - Handles all OpenGL calls.  GLX and OpenGL initializations are done at thread start (i.e., at OpenGLThread::preRun, not at the constructor)
 * - Creates an OpenGLFrameFifo instance (OpenGLThread::infifo) that has direct memory access to the GPU.  It is retrieved by calling OpenGLThread::getFifo (fifo's frames get initialized once OpenGLThread starts running)
 * - The main execution loop (OpenGLThread::run) reads OpenGLThread::infifo, passes frames to the presentation queue OpenGLThread::presfifo and presents the frames accordint to their presentation timestamps
 * 
 * See also: \ref pipeline
 * 
 * @ingroup threading_tag
 * @ingroup openglthread_tag
 */
class OpenGLThread : public Thread { // <pyapi>

public: // <pyapi>
  /** Default constructor
   * 
   * @param name          A name indentifying this thread
   * @param fifo_ctx      Parameters for the internal OpenGLFrameFifo
   * @param msbuftime     Jitter buffer size in milliseconds (default=100 ms)
   * 
   */
  OpenGLThread(const char* name, OpenGLFrameFifoContext fifo_ctx=OpenGLFrameFifoContext(), unsigned msbuftime=DEFAULT_OPENGLTHREAD_BUFFERING_TIME, const char* x_connection=""); // <pyapi>
  virtual ~OpenGLThread(); ///< Virtual destructor // <pyapi>
  
protected: // initialized at init list
  OpenGLFrameFifo   *infifo;      ///< This thread reads from this communication fifo
  FifoFrameFilter   infilter;     ///< A FrameFilter for writing incoming frames
  std::string       x_connection; ///< X-server connection string (i.e. ":0.0", ":0.1", etc.    
    
protected: // Shaders. Initialized by makeShaders. 
  // Future developments: create a shader instance (and a program) per each stream, etc..?
  // these could be reserved into the stack right here
  // YUVShader   yuv_shader;
  // RGBShader   rgb_shader;
  // .. but, we want to do all OpenGL calls after differentiating the Thread
  YUVShader*    yuv_shader; ///< Initialized by OpenGLThread::makeShaders
  RGBShader*    rgb_shader; ///< Initialized by OpenGLThread::makeShaders
  
protected: 
  YUVTEX*       statictex;           ///< Texture to be shown when there is no stream
  std::string   static_texture_file; ///< Name of the file where statictex is
  
protected: // Variables related to X11 and GLX.  Initialized by initGLX.
  Display*      display_id;
  bool          doublebuffer_flag;
  GLXContext    glc;
  int*          att;
  Window        root_id;
  XVisualInfo*  vi;
  GLXFBConfig*  fbConfigs;
  Colormap      cmap;
  YUVFrame*     dummyframe; ///< A PBO membuf which we reserve from the GPU as the first membuf, but is never used
  
  std::vector<SlotContext>              slots_;        ///< index => SlotContext mapping (based on vector indices)
  // std::map<SlotNumber, SlotContext>     slots_;        ///< SlotNumber => SlotContext mapping TODO: start using this in the future
  std::map<Window, RenderGroup>         render_groups; ///< window_id => RenderGroup mapping.  RenderGroup objects are warehoused here.
  std::vector<std::list<RenderGroup*>>  render_lists;  ///< Each vector element corresponds to a slot.  Each list inside a vector element has pointers to RenderGroups to be rendered.
  // std::map<SlotNumber,std::list<RenderGroup*>>  render_lists;  ///< TODO: start using this in the future
  
private: // function pointers for glx extensions
  PFNGLXSWAPINTERVALEXTPROC     pglXSwapIntervalEXT;
  PFNGLXGETSWAPINTERVALMESAPROC pglXGetSwapIntervalMESA;
  PFNGLXSWAPINTERVALMESAPROC    pglXSwapIntervalMESA;
  
public: // methods for getting/setting the swap interval .. these choose the correct method to call
  unsigned                      swap_flavor;
  unsigned                      swap_interval_at_startup;                             ///< The value of swap interval when this thread was started
  unsigned                      getSwapInterval(GLXDrawable drawable=0);
  void                          setSwapInterval(unsigned i, GLXDrawable drawable=0);
  
protected: // Variables related to queing and presentation
  unsigned             msbuftime;            ///< Buffering time in milliseconds
  unsigned             future_ms_tolerance;  ///< If frame is this much in the future, it will be discarded.  See OpenGLThread::OpenGLThread for its value (some multiple of OpenGLThread::msbuftime)
  std::list<Frame*>    presfifo;             ///< Double-linked list of buffered frames about to be presented
  long int             calltime, callswaptime;             ///< Debugging: when handleFifo was last called?
  
protected: // debugging variables
  bool  debug;
    
public: // manipulate RenderGroup(s)
  bool                hasRenderGroup(Window window_id); ///< Check if we have render groups
  RenderGroup&        getRenderGroup(Window window_id); ///< Returns a reference to RenderGroup in OpenGLThread::render_groups .. returns null if not found
  bool                newRenderGroup(Window window_id); ///< Creates a RenderGroup and inserts it into OpenGLThread::render_groups
  bool                delRenderGroup(Window window_id); ///< Remove RenderGroup from OpenGLThread::render_groups
  
public: // manipulate RenderContex(es)
  // int                 newRenderContext(SlotNumber slot, Window window_id, unsigned int z); ///< Creates a new render context
  void                newRenderContext(SlotNumber slot, Window window_id, int id, unsigned int z); ///< Creates a new render context
  bool                delRenderContext(int id); ///< Runs through OpenGLThread::render_groups and removes indicated RenderContext
  
public: // loading and rendering actions
  void                loadYUVFrame(SlotNumber n_slot, YUVFrame *yuvframe); ///< Load PBO to texture in OpenGLThread::slots_[n_slot]
  void                render(SlotNumber n_slot); ///< Render all RenderGroup(s) depending on slot n_slot
  void                checkSlots(long int mstime);
  
public: // getter methods
  Display*            getDisplayId() {return display_id;}
  const GLXContext&   getGlc() {return glc;}
  const Window&       getRootId() {return root_id;}

  
//public:
//  OpenGLFrameFifo&    getFifo();
  
public: // slot methods
  bool                slotUsed       (SlotNumber i);
  void                activateSlot   (SlotNumber i, YUVFrame *yuvframe);
  void                activateSlotIf (SlotNumber i, YUVFrame *yuvframe); // Activate slot if it's not already active or if the texture has changed
  bool                manageSlotTimer(SlotNumber i, long int mstime);
  
public: // setter methods
  void                debugOn()  {debug=true;}
  void                debugOff() {debug=false;}
  
public: // Thread variables
  std::deque<OpenGLSignalContext> signal_fifo;   ///< Redefinition of signal fifo.  Signal fifo of Thread::SignalContext(s) is now hidden.
  
public: // Thread virtual methods
  void run();                                       ///< Main execution loop is defined here.
  void preRun();                                    ///< Called before entering the main execution loop, but after creating the thread.  Calls OpenGLThread::createShaders and OpenGLThread::reserveFrames
  void postRun();                                   ///< Called after the main execution loop exits, but before joining the thread
  void handleSignal(OpenGLSignalContext &signal_ctx);
  void handleSignals();                             ///< From signals to methods 
  void sendSignal(OpenGLSignalContext signal_ctx);        ///< Send a signal to the thread 
  void sendSignalAndWait(OpenGLSignalContext signal_ctx); ///< Send a signal to the thread and wait all signals to be executed
  
public: // methods, internal : initializing / closing .. but we might want to test these separately, keep public
  void initGLX();          ///< Connect to X11 server, init GLX direct rendering
  void closeGLX();         ///< Close connection to X11
  void loadExtensions();   ///< Load OpenGL extensions using GLEW
  void VSyncOff();         ///< Turn off vsync for swapbuffers
  int hasCompositor(int screen); ///< Detect if a compositor is running
  void makeShaders();      ///< Compile shaders
  void delShaders();       ///< Delete shader
  void reserveFrames();    ///< Attaches YUVPBO instances with direct GPU memory access to Frame::yuvpbo
  void releaseFrames();    ///< Deletes YUVPBO by calling Frame::reset
  
protected: // internal methods
  void dumpPresFifo();  ///< Dump the presentation queue
  void diagnosis();     ///< Dump presentation queue size and diagnosis output for infifo (YUVdiagnosis)
  void resetCallTime();
  void reportCallTime(unsigned i);        ///< How much time since handleFifo exited
  long unsigned insertFifo(Frame* f);     ///< Sorted insert: insert a timestamped frame into the fifo
  void readStaticTex();                   ///< Reads a static texture that's shown on a window when no stream is received.  Uses OpenGLThread::static_texture_file
  
  /** Runs through the fifo, presents / scraps frames, returns timeout until next frame presentation
   * 
   * See also \ref timing
   */
  long unsigned handleFifo();
  void delRenderContexes();
  
public:  // reporting
  void reportSlots();
  void reportRenderGroups();
  void reportRenderList();
  
  void dumpYUVStacks() {infifo->dumpYUVStacks();} ///< State of the YUV Frame stack
  void YUVdiagnosis()  {infifo->YUVdiagnosis();}  ///< Brief resumen of the state of the YUV Frame stack
  void dumpInFifo()    {infifo->dumpFifo();}      ///< Incoming fifo: here we have all kinds of frames, including YUVFrame(s)
  
public: // testing 
  void recycle(Frame* f)             {infifo->recycle(f);}               ///< Recycle a frame back to OpenGLFrameFifo
  
public: // for testing // <pyapi>
  Window     createWindow(bool map=true, bool show=false);          ///< Creates a new X window (for debugging/testing only) // <pyapi>
  void       makeCurrent(const Window window_id);  ///< Set current X windox                                // <pyapi>
  unsigned   getVsyncAtStartup();                  ///< What vsync was at startup time?                     // <pyapi> 
  void       reConfigWindow(Window window_id);
  Window     getChildWindow(Window parent_id);
  
public: // API // <pyapi>
  FifoFrameFilter &getFrameFilter();                           ///< API method: get filter for passing frames to this thread // <pyapi>
  void setStaticTexFile(const char* fname);                    ///< API method: set a file where the static texture (yuv image that's shown when no stream is received) is read from // <pyapi>
  
  /** API call: stops the thread
   */
  void stopCall();                                             // <pyapi>
  
  /** API call: reports render groups and lists
   */
  void infoCall();                                             // <pyapi>
  
  /** API call: create a rendering group
   * @param window_id  X window id
   */
  bool newRenderGroupCall  (Window window_id);                 // <pyapi>
  
  /** API call: delete a rendering group
   * @param window_id  X window id
   */
  bool delRenderGroupCall  (Window window_id);                 // <pyapi>
  
  /** API call: create a new rendering context, i.e. a new stream-to-render mapping
   * @param slot       Slot number identifying the stream
   * @param window_id  X window id (corresponding to a created rendering group)
   * @param z          Stacking order of the rendered bitmap
   * 
   * returns a unique integer id presenting the rendering context.
   * return 0 if the call failed.
   * 
   */
  int newRenderContextCall (SlotNumber slot, Window window_id, unsigned int z);  // <pyapi>
  
  /** API call: delete a rendering context
   * @param id         Rendering context id
   * 
   */
  bool delRenderContextCall(int id);                           // <pyapi>
}; // <pyapi>

#endif
