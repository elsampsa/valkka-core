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
 *  @version 0.2.0 
 *  
 *  @brief The OpenGL thread for presenting frames and related data structures
 *
 */ 

#include "opengl.h"
#include "sizes.h"
#include "threads.h"
#include "shaders.h"


/** A FrameFifo with multiple stacks.  Used by OpenGLThread
 * 
 * This is a special FrameFifo class.  You should never instantiate it by yourself!  Instead, request it from OpenGLThread by calling OpenGLThread::getFifo (fifo's frames get properly initialized after OpenGLThread has been started and/or if OpenGLThread::preRun is called)
 *
 * Why not instantiate it by the API user?  The reason is simple: in order to construct an instance of OpenGLFrameFifo, we need to call various OpenGL routines, and in this library, all OpenGL calls are done by one, and only one thread, e.g. OpenGLThread.  
 * 
 * In detail, the pre-reserved frames we are warehousing in this object's stacks, have their Frame::frametype set to FrameType::yuvframe, and they are making use of the Frame::yuvpbo pointer.  Frame::yuvpbo are of the type YUVPBO, i.e. they present OpenGL Pixel Buffer Objects (PBOs), having direct memory access to the GPU memory.  And initializing OpenGL and reserving PBOs is done by the OpenGLThread.
 * 
 * In a typical situation, frames from AVThread are feeded, through FrameFilter(s) into OpenGLFrameFifo.  Decoded frames have typically have their Frame::frametype set to FrameType::avframe, indicating that the decoded frame is pointer by the Frame::av_frame pointer.  In such cases OpenGLFrameFifo then OpenGLFrameFifo::prepareAVFrame to upload the data from Frame::av_frame directly into GPU, using the %PBO(s).
 *  
 * @ingroup openglthread_tag
 */
class OpenGLFrameFifo : public FrameFifo { // <pyapi>
  
friend class OpenGLThread; // can manipulate reservoirs, stacks, etc.
  
public: // <pyapi>
  /** Default constructor
   * 
   * @param n_stack_720p  Size of the frame reservoir for 720p
   * @param n_stack_1080p Size of the frame reservoir for 1080p
   * @param n_stack_1440p Size of the frame reservoir for 1440p
   * @param n_stack_4K    Size of the frame reservoir for 4K
   * @param n_stack_audio Size of the payload reservoir for audio frames
   * 
   */
  OpenGLFrameFifo(unsigned short n_stack_720p, unsigned short n_stack_1080p, unsigned short n_stack_1440p, unsigned short n_stack_4K, unsigned short n_stack_audio); // <pyapi>
  /** Default destructor
   */
  ~OpenGLFrameFifo(); // <pyapi>
  
public:
  Frame* getFrame_(BitmapType bmtype);               ///< Take a frame from a stack.  Not mutex protection.  Only for internal use
  void reportStacks_();                              ///< Show stack usage.  No mutex protection
  void dumpStack_();                                 ///< Dump the frames in the stack.  No mutex protection
  
public: // mutex protected calls .. be carefull when calling mutex protected call inside a mutex protected call (always check mutex protected context)
  Frame* getAudioFrame();                            ///< Take a frame from the OpenGLFrameFifo::stack_audio stack
  Frame* prepareFrame(Frame* frame);                 ///< Create a copy of frames other than FrameType::avframe
  Frame* getFrame(BitmapType bmtype);                ///< Take a frame from a stack by bitmap type (uses OpenGLFrameFifo::getFrame_)
  Frame* prepareAVFrame(Frame* frame);               ///< Chooses a frame from the stack, does GPU uploading for frames of the type FrameType::avframe
  bool writeCopy(Frame* f, bool wait=false);         ///< Take a frame "ftmp" from a relevant stack, copy contents of "f" into "ftmp" (or do GPU uploading) and insert "ftmp" into the beginning of the fifo (i.e. perform "copy-on-insert").
  void recycle(Frame* f);                            ///< Return Frame f back into the stack (relevant stack is chosen automatically)
  void reportStacks();                               ///< Show stack usage
  void dumpStack();                                  ///< Dump the frames in the stack
  
public: // setters
  void debugOn() {debug=true;}
  void debugOff(){debug=false;}
  
private:
  bool               debug;
  unsigned short     n_stack_720p,n_stack_1080p, n_stack_1440p, n_stack_4K, n_stack_audio;
  std::deque<Frame*> stack_720p,  stack_1080p,   stack_1440p,   stack_4K,   stack_audio;
  std::vector<Frame> reservoir_720p, reservoir_1080p, reservoir_1440p, reservoir_4K, reservoir_audio;
}; // <pyapi>



/** Each Frame carries information about it's slot number in Frame::slot.  Slot number identifies the origin of the stream.  When the bitmap Frame arrives to OpenGLThread, each stream/slot requires:
 * - A set of OpenGL Textures (SlotContext::yuvtex)
 * - A reference to the relevant shader program (SlotContext::shader)
 * This class encapsulates all that.  SlotContext can be in an active (with textures reserved and shaders set) or in an inactive state.
 * 
 * @ingroup openglthread_tag
 */
class SlotContext {
 
public:
  SlotContext();  ///< Default constructor
  ~SlotContext(); ///< Default destructor
  
public:  
  YUVTEX*      yuvtex;  ///< This could be a base class for different kinds of textures (now only YUVTEX)
  YUVShader*   shader;  ///< Base class for the chosen Shader for this slot.  Now always YUVShader
  
private:
  bool      active;     ///< Is acticated or not
  AVCodecID codec_id  ; ///< FFmpeg codec id
  
public:
  void activate(GLsizei w, GLsizei h, YUVShader* shader);   ///< Allocate SlotContext::yuvtex (for a frame of certain size) and SlotContext::shader.
  void deActivate();         ///< Deallocate textures
  void loadTEX(YUVPBO* pbo); ///< Load bitmap from PBO to SlotContext::yuvtex
  
public: // getters
  bool isActive() const {return active;}   ///< Check if active
  
};



/** Encapsulates data for rendering a single bitmap: vertex array object (VAO), vertex buffer object (VBO), vertice coordinates, transformation matrix, etc.
 * 
 * @ingroup openglthread_tag
 * 
 */
class RenderContext {

public:
  /** Default constructor
   * 
   * @param slot_context A reference to a SlotContext instance
   * @param z            Stacking order (for "video-in-video")
   * 
   */
  RenderContext(const SlotContext& slot_context, unsigned int z=0);
  virtual ~RenderContext(); ///< Default virtual destructor
  
public: // Initialized at constructor init list or at constructor
  const SlotContext& slot_context; ///< Knows relevant Shader and YUVTEX
  const unsigned int z;            ///< Stacking order
  bool active;
  int id;                          ///< A unique id identifying this render context
  
public: // Initialized at RenderContext::init
  GLuint        VAO;     ///< id of the vertex array object
  GLuint        VBO;     ///< id of the vertex buffer object
  GLuint        EBO;     ///< id of the element buffer object
  std::array<GLfloat,16> transform; ///< data of the transformation matrix
  std::array<GLfloat,20> vertices;  ///< data of the vertex buffer object
  std::array<GLuint, 6>  indices;   ///< data of the element buffer object
  
public:
  XWindowAttributes x_window_attr;  ///< X window size, etc.
  
public: // getters
  int getId() const {return id;}    ///< Returns the unique id of this render context
  bool isActive() {return active;}
  
  
public:
  bool activate();        ///< Activate this RenderContext (now it can render)
  bool activateIf();      ///< Try to activate if not active already
  void render(XWindowAttributes x_window_attr);  ///< Does the actual drawing of the bitmap
  void bindTextures();    ///< Associate textures with the shader program.  Uses parameters from Shader reference.
  void bindVars();        ///< Upload other data to the GPU (say, transformation matrix).  Uses parameters from Shader reference.
  void bindVertexArray(); ///< Bind the vertex array and draw
};

// https://stackoverflow.com/questions/4421706/what-are-the-basic-rules-and-idioms-for-operator-overloading
// inline bool operator==(const RenderContext& lhs, const RenderContext& rhs){ return (lhs.getId()==rhs.getId()); }


/** Group of bitmaps that are rendered into the same X-window.  A RenderGroup instance is indentified uniquely by an x-window id.  API user should **never** create this (use the API calls in OpenGLThread instead)
 * 
 * @ingroup openglthread_tag
 * 
 */ 
class RenderGroup {
  
public:
  /** Default constructor
   * 
   * @param display_id        Pointer to x display id
   * @param glc               Reference to GXLContext
   * @param window_id         X window id
   * @param doublebuffer_flag Use double buffering or not (default true)
   */
  RenderGroup(Display *display_id, const GLXContext &glc, Window window_id, bool doublebuffer_flag=true);
  ~RenderGroup();        ///< Default destructor
  
public:
  Display* display_id;    ///< X display id
  const GLXContext &glc;  ///< GLX Context
  Window   window_id;     ///< X window id
  bool     doublebuffer_flag; ///< Double buffer rendering or not?
  std::list<RenderContext> render_contexes; ///< RenderContext instances in ascending z order.  User created rendercontexes are warehoused here
  XWindowAttributes x_window_attr;          ///< X window attributes
  
public: // getters
  Window getWindowId() const {return window_id;}
  const std::list<RenderContext>& getRenderContexes() const {return render_contexes;}
  
  
public:
  std::list<RenderContext>::iterator getContext(int id);
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



/** Signal information for OpenGLThread
 * 
 * @ingroup openglthread_tag
 */
struct OpenGLSignalContext {   // used by signals:                                
  SlotNumber    n_slot;        ///< in: new_render_context                                         
  Window        x_window_id;   ///< in: new_render_context, new_render_group, del_render_group     
  unsigned int  z;             ///< in: new_render_context                                         
  int           render_ctx;    ///< in: del_render_context, out: new_render_context                
  bool          success;       ///< return value: was the call succesful?                          
};                                                                                               


std::ostream &operator<<(std::ostream &os, OpenGLSignalContext const &m);


/** This class does a lot of things:
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
   * @param n_stack_720p  Size of the frame reservoir for 720p
   * @param n_stack_1080p Size of the frame reservoir for 1080p
   * @param n_stack_1440p Size of the frame reservoir for 1440p
   * @param n_stack_4K    Size of the frame reservoir for 4K
   * @param n_stack_audio Size of the payload reservoir for audio frames
   * @param msbuftime     Jitter buffer size in milliseconds (default=100 ms)
   * @param core_id       Force thread affinity (default=-1 = no affinity)
   * 
   * n_stack parameters are used to initialize OpenGLFrameFifo.  When OpenGLThread is started (at OpenGLThread::preRun), OpenGLThread::reserveFrames is called.  OpenGLThread::reserveFrames then instantiates YUVPBO objects, that have direct memory access to GPU.  YUVPBO instances are attached to the Frame::yuvpbo pointer of frames in OpenGLFrameFifo's frame reservoirs.
   * 
   */
  OpenGLThread(const char* name, unsigned short n720p=0, unsigned short n1080p=0, unsigned short n1440p=0, unsigned short n4K=0, unsigned short naudio=0, unsigned msbuftime=100, int core_id=-1); // <pyapi>
  virtual ~OpenGLThread(); ///< Virtual destructor // <pyapi>
  
public:
  /** List of possible signals for the thread
   */
  enum class Signals {
    none,                 ///< null signal
    exit,                 ///< exit
    info,                 ///< used by API infoCall
    new_render_group,     ///< used by API newRenderCroupCall
    del_render_group,     ///< used by API delRenderGroupCall
    new_render_context,   ///< used by API newRenderContextCall
    del_render_context    ///< used by API delRenderContextCall
  };
  /** Encapsulates data sent by the signal
   * 
   * Has the enumerated signal from OpenGLThread::Signals class plus any other necessary data, represented by OpenGLSignalContext.
   */
  struct SignalContext {
    Signals              signal; ///< The signal
    OpenGLSignalContext  *ctx;   ///< Why pointers? .. we have return values here
  };

protected: // initialized at init list
  OpenGLFrameFifo     infifo; ///< This thread reads from this communication fifo
    
protected: // Shaders. Initialized by makeShaders. 
  // Future developments: create a shader instance (and a program) per each stream, etc..?
  // these could be reserved into the stack right here
  // YUVShader   yuv_shader;
  // RGBShader   rgb_shader;
  // .. but, we want to do all OpenGL calls after differentiating the Thread
  YUVShader*    yuv_shader; ///< Initialized by OpenGLThread::makeShaders
  RGBShader*    rgb_shader; ///< Initialized by OpenGLThread::makeShaders
  
protected: // Variables related to X11 and GLX.  Initialized by initGLX.
  Display*      display_id;
  bool          doublebuffer_flag;
  GLXContext    glc;
  int*          att;
  Window        root_id;
  XVisualInfo*  vi;
  GLXFBConfig*  fbConfigs;
  Colormap      cmap;
  
  std::vector<SlotContext>              slots_;        ///< index => SlotContext mapping (based on vector indices)
  std::map<Window, RenderGroup>         render_groups; ///< window_id => RenderGroup mapping.  RenderGroup objects are warehoused here.
  std::vector<std::list<RenderGroup*>>  render_lists;  ///< Each vector element corresponds to a slot.  Each list inside a vector element has pointers to RenderGroups to be rendered.
  
protected: // Variables related to queing and presentation
  unsigned             msbuftime;            ///< Buffering time in milliseconds
  unsigned             future_ms_tolerance;  ///< If frame is this much in the future, it will be discarded.  See OpenGLThread::OpenGLThread for its value (some multiple of OpenGLThread::msbuftime)
  std::list<Frame*>    presfifo;             ///< Double-linked list of buffered frames about to be presented
  
protected: // debugging variables
  bool  debug;
    
public: // manipulate RenderGroup(s)
  bool                hasRenderGroup(Window window_id); ///< Check if we have render groups
  RenderGroup&        getRenderGroup(Window window_id); ///< Returns a reference to RenderGroup in OpenGLThread::render_groups .. returns null if not found
  bool                newRenderGroup(Window window_id); ///< Creates a RenderGroup and inserts it into OpenGLThread::render_groups
  bool                delRenderGroup(Window window_id); ///< Remove RenderGroup from OpenGLThread::render_groups
  
public: // manipulate RenderContex(es)
  int                 newRenderContext(SlotNumber slot, Window window_id, unsigned int z); ///< Creates a new render context
  bool                delRenderContext(int id); ///< Runs through OpenGLThread::render_groups and removes indicated RenderContext
  
public: // loading and rendering actions
  void                loadTEX(SlotNumber n_slot, YUVPBO* pbo); ///< Load PBO to texture in slot n_slot
  void                render(SlotNumber n_slot); ///< Render all RenderGroup(s) depending on slot n_slot
  
public: // getter methods
  Display*            getDisplayId() {return display_id;}
  const GLXContext&   getGlc() {return glc;}
  const Window&       getRootId() {return root_id;}
  
public: // API // <pyapi>
  OpenGLFrameFifo&    getFifo(); ///< Retrieve the communication fifo  // <pyapi>
  
public: // setter methods
  void                activateSlot   (SlotNumber i, BitmapType bmtype);
  void                activateSlotIf (SlotNumber i, BitmapType bmtype); // activate if not already active
  void                debugOn()  {debug=true;}
  void                debugOff() {debug=false;}
  
public: // Thread variables
  std::deque<SignalContext> signal_fifo;   ///< Redefinition of signal fifo.  Signal fifo of Thread::SignalContext(s) is now hidden.
  
public: // Thread virtual methods
  void run();     ///< Main execution loop is defined here. \callgraph
  void preRun();  ///< Called before entering the main execution loop, but after creating the thread \callgraph
  void postRun(); ///< Called after the main execution loop exits, but before joining the thread \callgraph
  void handleSignals(); ///< From signals to methods \callgraph
  void sendSignal(SignalContext signal_ctx); ///< Send a signal to the thread \callgraph
  void sendSignalAndWait(SignalContext signal_ctx); ///< Send a signal to the thread and wait all signals to be executed \callgraph
  
public: // methods, internal : initializing / closing .. but we might want to test these separately, keep public
  void initGLX();          ///< Connect to X11 server, init GLX direct rendering
  void closeGLX();         ///< Close connection to X11
  void loadExtensions();   ///< Load OpenGL extensions using GLEW
  void makeShaders();      ///< Compile shaders
  void delShaders();       ///< Delete shader
  void reserveFrames();    ///< Attaches YUVPBO instances with direct GPU memory access to Frame::yuvpbo \callgraph
  void releaseFrames();    ///< Deletes YUVPBO by calling Frame::reset \callgraph
  
protected: // internal methods
  void dumpFifo();
  long unsigned insertFifo(Frame* f);     ///< Sorted insert: insert a timestamped frame into the fifo \callgraph
  
  /** Runs through the fifo, presents / scraps frames, returns timeout until next frame presentation \callgraph
   * 
   * See also \ref timing
   * 
   */
  long unsigned handleFifo();
  void delRenderContexes();
  
public:  // reporting
  void reportSlots();
  void reportRenderGroups();
  void reportRenderList();
  
  void reportStacks() {infifo.reportStacks();}
  void dumpStack()    {infifo.dumpStack();}
  void dumpInFifo()   {infifo.dumpFifo();}
  
public: // testing 
  Frame* getFrame(BitmapType bmtype) {return infifo.getFrame(bmtype);} ///< Get a frame from the OpenGLFrameFifo
  void recycle(Frame* f)             {infifo.recycle(f);} ///< Recycle a frame back to OpenGLFrameFifo
  
public: // for testing // <pyapi>
  Window createWindow();   ///< Creates a new X window (for debugging/testing only) // <pyapi>
  void makeCurrent(const Window window_id);  ///< Set current X windox // <pyapi>
  
public: // API // <pyapi>
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

