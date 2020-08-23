/*
 * openglthread.cpp : The OpenGL thread for presenting frames and related data structures
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
 *  @file    openglthread.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.18.0 
 *  
 *  @brief The OpenGL thread for presenting frames and related data structures
 *
 *  @section DESCRIPTION
 *  
 */ 

#include "openglthread.h"
#include "tools.h"
#include "logging.h"

// WARNING: these define switches should be off (commented) by default
// #define PRESENT_VERBOSE 1 // enable this for verbose output about queing and presenting the frames in OpenGLThread // @verbosity       
// #define RENDER_VERBOSE 1 // enable this for verbose rendering
// #define NO_LATE_DROP_DEBUG 1 // don't drop late frame, but present everything in OpenGLThreads fifo.  Useful when debuggin with valgrind (as all frames arrive ~ 200 ms late)

// PFNGLXSWAPINTERVALEXTPROC glXSwapIntervalEXT = (PFNGLXSWAPINTERVALEXTPROC)glXGetProcAddress((const GLubyte*)"glXSwapIntervalEXT");  // Set the glxSwapInterval to 0, ie.

typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);


SlotContext::SlotContext(YUVTEX *statictex, YUVShader* shader) : statictex(statictex), shader(shader), yuvtex(NULL), active(false), codec_id(AV_CODEC_ID_NONE), bmpars(BitmapPars()), lastmstime(0), is_dead(true), ref_count(0), load_flag(false), keep_flag(false) {
}


SlotContext::~SlotContext() {
    deActivate(); 
}


// void SlotContext::activate(BitmapPars bmpars, YUVShader* shader) {//Allocate SlotContext::yuvtex and SlotContext::shader
void SlotContext::activate(BitmapPars bmpars) {
    if (active) {
        deActivate();
    }
    this->bmpars =bmpars;
    // this->shader =shader;
    opengllogger.log(LogLevel::crazy) << "SlotContext: activate: activating for w, h " <<bmpars.width << " " << bmpars.height << " " << std::endl;
    yuvtex=new YUVTEX(bmpars); // valgrind_debug protected
    active=true;
    is_dead=true;
    //load_flag = false; // nopes
    //keep_flag = false; // nopes
}


void SlotContext::deActivate() {//Deallocate
    active=false;
    load_flag = false;
    keep_flag = false;
    if (yuvtex!=NULL) {delete yuvtex;}
    yuvtex=NULL;
    bmpars=BitmapPars();
    // if (shader!=NULL) {delete shader;}
    // shader=NULL;
}


void SlotContext::loadYUVFrame(YUVFrame *yuvframe) {
    // return;
    #ifdef PRESENT_VERBOSE
    std::cout << "SlotContext: loadYUVFrame: "<< *yuvframe <<std::endl;
    #endif
    #ifdef OPENGL_TIMING
    if (yuvframe->mstimestamp<=prev_mstimestamp) { // check that we have fed the frames in correct order (per slot)
        std::cout << "SlotContext: loadYUVFrame: feeding frames in reverse order!" << std::endl;
    }
    prev_mstimestamp=yuvframe->mstimestamp;
    #endif
    
    yuvtex->loadYUVFrame(yuvframe); // from pbo to texture.  has valgrind_gpu_debug 
    load_flag = true;
}


bool SlotContext::manageTimer(long int mstime) {
    if ( (mstime-lastmstime) <=10 ) {
        opengllogger.log(LogLevel::normal) << "OpenGLThread: handleFifo: feeding frames too fast, dropping, dt=" << (mstime-lastmstime) << std::endl; // if negative, frames would have been fed in reverse order
        lastmstime=mstime;
        return false;
    }
    else {
        // std::cout << "OpenGLThread: dt " << mstime-lastmstime << std::endl;
        lastmstime=mstime;
        return true;
    }
}


void SlotContext::checkIfDead(long int mstime) {
    // std::cout << "SlotContext: checkIfDead: mstime, lastmstime " << mstime << " " << lastmstime << std::endl;
    if ( (mstime-lastmstime) >= 2000 ) { // nothing received in 2 secs
        is_dead=true;
    }
    else {
        is_dead=false;
    }
}


bool SlotContext::isPending(long int mstime) {
    return ( (mstime-lastmstime) >= 100 ); // nothing received in 100 ms
}


void SlotContext::loadFlag(bool val) {
    load_flag = val;
}

void SlotContext::keepFlag(bool val) {
    keep_flag = val;
}



YUVTEX* SlotContext::getTEX() {
    /* So, with respect to the deadtime timeout, this can have two states:
     * - Keep on showing the last frame
     * - Show the static background image
     * 
     */
    
    if (!active) { // nothing much to show ..
        // std::cout << "OpenGLThread : SlotContext : getText : not active" << std::endl;
        return statictex;
    }
    
    // std::cout << "OpenGLThread : SlotContext : getText : load, keep, is_dead " << load_flag << " " << keep_flag << " " << is_dead << std::endl;
    
    if (load_flag) { // can use this yuvtex
        if (keep_flag) { // keep on showing the last frame, no matter what
            return yuvtex;
        }
        else if (is_dead) { // no frames have been received, and we are not forced to use the last frame
            return statictex;
        }
        else { // everything's ok .. stream is alive
            return yuvtex;
        }
    }
    else { // if load_flag is cleared, the current yuvtex must not be used.  Must wait for a new yuvtex
        return statictex;
    }
    
    /* // old turf
    if (is_dead) {
        return statictex;
    }
    else if (active) {
        return yuvtex;
    }
    */
}



// RenderContext::RenderContext(SlotContext &slot_context, YUVTEX *statictex, unsigned int z) : slot_context(slot_context), statictex(statictex), z(z), active(false) {
RenderContext::RenderContext(SlotContext *slot_context, int id, unsigned int z) : slot_context(slot_context), id(id), z(z) {
    // https://learnopengl.com/#!Getting-started/Textures
    // https://www.khronos.org/opengl/wiki/Vertex_Specification
    // https://gamedev.stackexchange.com/questions/86125/what-is-the-relationship-between-glvertexattribpointer-index-and-glsl-location
    // https://stackoverflow.com/questions/5532595/how-do-opengl-texture-coordinates-work
    // So, what is the vertex array object all about:
    // https://stackoverflow.com/questions/8704801/glvertexattribpointer-clarification
    
    /* // bad idea..
     *  struct timeval time;
     *  gettimeofday(&time, NULL);
     *  id=time.tv_sec*1000+time.tv_usec/1000;
     */
    // id = std::rand();

    render_mstime = 0;
    render_mstime_old = 0;

    slot_context->inc_ref_count();
    activate();
}


RenderContext::~RenderContext() {
    slot_context->dec_ref_count();
    clearObjects();
    // return;
    #ifdef VALGRIND_GPU_DEBUG
    #else
    // TODO: WARNING: we're managing a GPU resource here.  Know what you're doing
    
    ///*
    glDeleteVertexArrays(1, &VAO);
    // glDeleteBuffers(1, &VAO); // This made mysterious segfaults!
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    //*/
    // std::cout << "RenderContext: dtor" << std::endl;
    #endif
}


/*
 * bool RenderContext::activateIf() {
 * if (!active) { // not active .. try to activate
 *  return activate(); 
 * }
 * else {
 *   return true;
 * }
 * }
 */


void RenderContext::activate() { 
    /*
     *  if (!slot_context.isActive()) {
     *    return false; // could not activate..
}
// so, slot context has been properly initialized => we have Shader instance
active=true;
*/
    Shader* shader=slot_context->shader; // assume the shaders dont change (practically always YUV)
    struct timeval time;
    unsigned int transform_size, vertices_size, indices_size;
    
    transform =std::array<GLfloat,16>{
        1.0f,             0.0f,             0.0f,   0.0f, 
        0.0f,             1.0f,             0.0f,   0.0f,
        0.0f,             0.0f,             1.0f,   0.0f,
        0.0f,             0.0f,             0.0f,   1.0f
    };
    transform_size=sizeof(GLfloat)*transform.size();
    
    // Let's define the vertices!
    // If you look at the vertex shader programs, you'll see that ..
    // .. we just make a single transform with a 4d matrix
    // At RenderGroup::render there is glViewPort call.  This means that there will be one matrix multiplication by the OpenGL rendering pipeline
    // .. before that multiplication, the vertex coordinates must be in normalized device coordinates (NDCs), i.e. from -1 to 1 in all dimensions
    // see here: http://www.songho.ca/opengl/gl_transform.html
    
    /* normalized device coordinates
     *  vertices =std::array<GLfloat,20>{
     *    //Positions          Texture Coords
     *    //Shader class references:
     *    //"position"        "texcoord"
     *    1.0f,  1.0f, 0.0f,   1.0f, 1.0f, // Top Right
     *    1.0f, -1.0f, 0.0f,   1.0f, 0.0f, // Bottom Right
     *   -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, // Bottom Left
     *   -1.0f,  1.0f, 0.0f,   0.0f, 1.0f  // Top Left 
};
vertices_size=sizeof(GLfloat)*vertices.size();

indices =std::array<GLuint,6>{  // Note that we start from 0!
0, 1, 3, // First Triangle
1, 2, 3  // Second Triangle
};
*/
    
    /* fooling around ..
     *  // this version works .. order does not matter, as long there is correspondence between position and texcoord
     *  vertices =std::array<GLfloat,20>{
     *    //Positions          Texture Coords
     *    //Shader class references:
     *    //"position"        "texcoord"
     *    0.0f,  0.0f, 0.0f,   0.0f, 0.0f, // bottom left
     *    0.0f,  1.0f, 0.0f,   0.0f, 1.0f, // top left 
     *    1.0f,  1.0f, 0.0f,   1.0f, 1.0f, // top right
     *    1.0f,  0.0f, 0.0f,   1.0f, 0.0f  // bottom right
};
vertices_size=sizeof(GLfloat)*vertices.size();
*/
    
    // normalized device coordinates and correct ffff%&#!!! flipping of the image
    vertices =std::array<GLfloat,20>{
        // Positions          Texture Coords
        // Shader class references:
        // "position"        "texcoord"
        -1.0f, -1.0f, 0.0f,   0.0f, 1.0f, // bottom left
        -1.0f,  1.0f, 0.0f,   0.0f, 0.0f, // top left 
         1.0f,  1.0f, 0.0f,   1.0f, 0.0f, // top right
         1.0f, -1.0f, 0.0f,   1.0f, 1.0f  // bottom right
    };
    vertices_size=sizeof(GLfloat)*vertices.size();
    
    indices =std::array<GLuint,6>{  // Note that we start from 0!
        0, 1, 2, // First Triangle
        2, 3, 0  // Second Triangle
    };
    
    indices_size=sizeof(GLuint)*indices.size();
    
    // std::cout << "SIZEOF: " << sizeof(vertices) << " " << vertices_size << std::endl; // eh.. its the same
    #ifdef VALGRIND_GPU_DEBUG
    #else
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    
    opengllogger.log(LogLevel::crazy) << "RenderContext: activate: VAO, VBO, EBO " << VAO << " " << VBO << " " << EBO << std::endl;
    opengllogger.log(LogLevel::crazy) << "RenderContext: activate: position, texcoord " << shader->position << " " << shader->texcoord << " " << std::endl;
    
    glBindVertexArray(VAO); // VAO works as a "mini program" .. we do all the steps below, when binding the VAO
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices.data(), GL_STATIC_DRAW);
    glBufferData(GL_ARRAY_BUFFER, vertices_size, vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    // glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices.data(), GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_size, indices.data(), GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(shader->position, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0); // this refers to the "Positions" part of vertices
    // 0: shader prog ref, 3: three elements per vertex
    glEnableVertexAttribArray(shader->position); // this refers to (location=0) in the shader program
    // TexCoord attribute
    glVertexAttribPointer(shader->texcoord, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat))); // this refers to "Texture Coords" part of vertices
    // 1: shader prog ref, 2: two elements per vertex
    glEnableVertexAttribArray(shader->texcoord); // this refers to (location=1) in the shader program
    
    glBindVertexArray(0); // Unbind VAO
    #endif
    
    // return true;
}


void RenderContext::addRectangle(float left, float right, float top, float bottom) {
    opengllogger.log(LogLevel::debug) << "RenderContext: addRectangle: " << left << " " << right << " " << top << " " << bottom << std::endl;
    Rectangle *obj = new Rectangle();
    obj->setCoordinates(left, right, top, bottom);
    overlays.push_front(obj);
}
    
    
void RenderContext::clearObjects() {
    for (auto it=overlays.begin(); it!=overlays.end(); ++it) {
        delete *it;
    }
    overlays.clear();
}
    
    
void RenderContext::renderTexture() { // TODO
}
    
void RenderContext::renderObjects() {
    for (auto it=overlays.begin(); it!=overlays.end(); ++it) {
        (*it)->draw();
    }
}



void RenderContext::render(XWindowAttributes x_window_attr) {// Calls bindTextures, bindParameters and bindVertexArray
    Shader* shader=slot_context->shader;
    
    // in the old code:
    // shader->use() .. copy from pbo to tex .. makeCurrent .. glViewport, glClear .. bindVertex, drawelements, unbind vertex array
    // here:
    // handleFifo: loadTex: loadYUVTEX: copy from pbo to tex .. RenderGroup::render: makeCurrent .. glViewport, glClear .. RenderContext::render: shader->use() .. 
    
    #ifdef RENDER_VERBOSE
    std::cout << "RenderContext: render: " << std::endl;
    #endif
    #ifdef RENDER_VERBOSE
    std::cout << "RenderContext: render: rendering!" << std::endl;
    #endif
    
    #ifdef OPENGL_TIMING
    long int mstime = getCurrentMsTimestamp();
    long int swaptime = mstime;
    #endif
    
    shader->use(); // use the shader
    this->x_window_attr=x_window_attr;
    
    bindTextures(); // valgrind_debug protected
    
    #ifdef OPENGL_TIMING
    swaptime=mstime; mstime=getCurrentMsTimestamp();
    if ( (mstime-swaptime) > 2 ) {
        std::cout << "RenderContext: render : render timing       : " << mstime-swaptime << std::endl;
    }
    #endif
    
    bindVars(); // valgrind_debug protected
    
    #ifdef OPENGL_TIMING
    swaptime=mstime; mstime=getCurrentMsTimestamp();
    if ( (mstime-swaptime) > 2 ) {
        std::cout << "RenderContext: render : bindvars timing     : " << mstime-swaptime << std::endl;
    }
    #endif
    
    bindVertexArray(); // valgrind_debug protected
    
    #ifdef OPENGL_TIMING
    swaptime=mstime; mstime=getCurrentMsTimestamp();
    if ( (mstime-swaptime) > 2 ) {
        std::cout << "RenderContext: render : bindvertexarr timing: " << mstime-swaptime << std::endl;
    }
    #endif
    
    shader->use_obj(); // use the shader for drawing overlay objects
    bindVarsObj();
    renderObjects();

    render_mstime = getCurrentMsTimestamp();
    // std::cout << "render timing: " << render_mstime - render_mstime_old << std::endl;
    render_mstime_old = render_mstime;

}


void RenderContext::bindTextures() {// Associate textures with the shader program.  Uses parameters from Shader reference.
    YUVShader *shader = (YUVShader*)(slot_context->shader); // shorthand
    // const YUVTEX *yuvtex;
    const YUVTEX *yuvtex =slot_context->getTEX(); // returns the relevant texture (static or live)
    
    /*
     *  if (slot_context.isDead()) { // no frame has been received for a while .. show the static texture instead
     * #ifdef PRESENT_VERBOSE
     *    std::cout << "RenderContext: bindTextures: using static texture" << std::endl;
     * #endif
     *    yuvtex    = statictex;
}
else { // everything's ok, use the YUVTEX in the SlotContext
    yuvtex    = (YUVTEX*)(slot_context.yuvtex);
}

*/
    
    #ifdef RENDER_VERBOSE
    std::cout << "RenderContext: bindTextures: indices y, u, v = " << yuvtex->y_index <<" "<< yuvtex->u_index <<" "<< yuvtex->v_index << std::endl;
    std::cout << "RenderContext: bindTextures: shader refs     = " << shader->texy <<" "<< shader->texu <<" "<< shader->texv << std::endl;
    #endif
    
    // slot_context.loadTEX(); // not necessary
    
    #ifdef VALGRIND_GPU_DEBUG
    #else
    
    // TODO: so.. RenderContext uses SlotContext's yuvtex..  If no frames is reserved in N millisecons, SlotContext::yuvtex could point
    // to an auxiliary texture (with, say, the Valkka logo)
    // that aux texture could be a static member of the SlotContext class ..?
    // nopes .. use method SlotContext::setStaticTEX
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, yuvtex->y_index);
    glUniform1i(shader->texy, 0); // pass variable to shader
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, yuvtex->u_index);
    glUniform1i(shader->texu, 1); // pass variable to shader
    
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, yuvtex->v_index);
    glUniform1i(shader->texv, 2); // pass variable to shader
    
    #endif
}


void RenderContext::bindVars() {// Upload other data to the GPU (say, transformation matrix).  Uses parameters from Shader reference.
    // eh.. now we're doing this on each "sweep" .. we could give a callback to RenderGroup that would do the uploading of transformatio matrix
    // .. on the other hand, its not that much data (compared to bitmaps) !
    YUVShader *shader = (YUVShader*)(slot_context->shader); // shorthand
    // const YUVTEX *yuvtex;
    const YUVTEX *yuvtex =slot_context->getTEX(); // returns the relevant texture (static or live)
    
    XWindowAttributes& wa=x_window_attr; // shorthand
    // GLfloat r, dx, dy; // , dx2, dy2;
    GLfloat r, kx, ky;
    
    // calculate dimensions
    // (screeny/screenx) / (iy/ix)  =  screeny*ix / screenx*iy
    
    const BitmapPars &bmpars =yuvtex->bmpars; // pre-reserved bitmap dimensions
    // const BitmapPars &source_bmpars =yuvtex->source_bmpars; // actual bitmap dimensions
    
    // std::cout << "RenderContext: bindVars: w0, h0 = " << bmpars.width <<", "<< bmpars.height<<" "<< std::endl;
    // std::cout << "RenderContext: bindVars: w, h   = " << source_bmpars.width <<", "<<source_bmpars.height<<" "<< std::endl;
    
    /* glViewPort expects everything to be in the L scale (where L is the scale of the reserved bitmap)
     *  
     *  NOTE: Actually, we don't need this if we keep the image and bitmap size the same (which is the only sane way of doing this..)
     *  
     *  __________L_________
     *  _l_
     *      ______d__________    
     *  xxxx----------------.
     *                      (scale center)  
     *  
     *  ----- = reserved bitmap
     *  xxxxx = true bitmap (part of the reserved)
     *  L     = size of reserved bitmap
     *  l     = size of true bitmap
     *  d     = distance of true bitmap from the scale center
     *  
     *  Scale it with (l/L) => now true bitmap size will be the size of the reserved bitmap (just what glViewport wants)
     *  
     *  d = (L-l)*(L/l) = (L/l-1) L
     *  
     *  In scale (expected by glViewPort) of L that's (L/l-1)
     */
    
    /* Scaling in general ..
     *  fixed scaling     unscale       desired scale
     *  [win.w/bm.w]   * (bm.w/win.w) * k
     *  
     *  We can't change the fixed scaling part (that's dictated by OpenGLViewPort
     */
    
    // L/l
    
    /*
     *  dx=float(bmpars.width) /float(source_bmpars.width);
     *  dy=float(bmpars.height)/float(source_bmpars.height);
     *  
     *  transform[12]=dx-1.0;
     *  transform[13]=-dy+1.0;
     */
    // std::cout << "RenderContext: bindVars: dx, dy = " << dx << ", " << dy << std::endl;
    
    
    // keep aspect ratio, clip image.  TODO: should be able to choose between clipping and black borders.
    //
    r=float(wa.width*(bmpars.height)) / float(wa.height*(bmpars.width));
    if (r<1.){ // screen form: wider than image // keep width, scale up height
        // kx=1; ky=r; // black borders
        kx=1/r; ky=1; // clipping
    }
    else if (r>1.) { // screen form: taller than image // keep height, scale up width
        // kx=1/r; ky=1; // black borders
        kx=1; ky=r; // clipping
    }
    else {
        kx=1;
        ky=1;
    }
    
    #ifdef RENDER_VERBOSE
    std::cout << "RenderContext: bindVars: kx, ky = " << kx <<" "<< ky<<" "<< std::endl;
    #endif  
    // /* // test..
    transform[0]  =kx;
    transform[5]  =ky;
    
    // */
    /*
     *  transform= {
     *    {dx                0.0f,             0.0f,   0.0f}, 
     *    {0.0f,             dy                0.0f,   0.0f},
     *    {0.0f,             0.0f,             1.0f,   0.0f},
     *    {0.0f,             0.0f,             0.0f,   1.0f}
};
*/
    #ifdef VALGRIND_GPU_DEBUG
    #else
    glUniformMatrix4fv(shader->transform, 1, GL_FALSE, transform.data());
    #endif
}

void RenderContext::bindVarsObj() {
    // transformation has been calculated .. no need to recalc
    YUVShader *shader = (YUVShader*)(slot_context->shader); // shorthand
    #ifdef VALGRIND_GPU_DEBUG
    #else
    glUniformMatrix4fv(shader->transform_obj, 1, GL_FALSE, transform.data());
    #endif
}


void RenderContext::bindVertexArray() {// Bind the vertex array and draw
    
    #ifdef RENDER_VERBOSE
    std::cout << "RenderContext: bindVertexArray: VAO= " << VAO << std::endl;
    #endif
    
    #ifdef TIMING_VERBOSE
    long int dt=getCurrentMsTimestamp();
    #endif
    
    #ifdef VALGRIND_GPU_DEBUG
    #else
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    #endif
    
    #ifdef TIMING_VERBOSE
    dt=getCurrentMsTimestamp()-dt;
    if (dt>10) {
        std::cout << "RenderContext: bindVertexArray : timing : drawing took " << dt << " ms" <<std::endl;
    }
    #endif
    
    
    #ifdef RENDER_VERBOSE
    std::cout << "RenderContext: bindVertexArray: " << std::endl;
    #endif
    
}

/*
 * void RenderContext::unBind() {
 *  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0); // unbind
 *  glBindTexture(GL_TEXTURE_2D, 0); // unbind 
 * }
 */



RenderGroup::RenderGroup(Display* display_id, const GLXContext& glc, Window window_id, Window child_id, bool doublebuffer_flag) : display_id(display_id), glc(glc), window_id(window_id), child_id(child_id), doublebuffer_flag(doublebuffer_flag) {
}


RenderGroup::~RenderGroup() {
}


std::list<RenderContext*>::iterator RenderGroup::getContext(int id) {
    std::list<RenderContext*>::iterator it;
    for(it=render_contexes.begin(); it!=render_contexes.end(); ++it) {
        if ((*it)->getId()==id) {
            return it;
        }
    }
    return it;
}


bool RenderGroup::addContext(RenderContext *render_context) {
    
    /*
     *  render_contexes.push_back(render_context); 
     *  return true;
     */
    
    ///*
    if (getContext(render_context->getId())==render_contexes.end()) {
        render_contexes.push_back(render_context); 
        return true;
    }
    else { // there is a RenderContext with the same id here..
        return false;
    }
    //*/
}


RenderContext* RenderGroup::delContext(int id) {
    std::list<RenderContext*>::iterator it = getContext(id);
    if (it==render_contexes.end()) {
        return NULL;
    }
    else {
        RenderContext* ctx =*it;
        render_contexes.erase(it); // this drives the compiler mad.. it starts moving stuff in the container..?
        // render_contexes.pop_back(); // this is ok..
        // return (*it); // can't dereference ? TODO
        return ctx;
    }
}


bool RenderGroup::isEmpty() {
    return render_contexes.empty();
}


void RenderGroup::render() {
    
    #ifdef PRESENT_VERBOSE
    std::cout << "RenderGroup: " << std::endl;
    std::cout << "RenderGroup: start rendering!" << std::endl;
    std::cout << "RenderGroup: render: display, window_id, child_id " <<display_id<<" "<<window_id<<" "<<child_id << std::endl;
    #endif
    
    // glFlush();
    // glFinish();
    
    #ifdef OPENGL_TIMING
    long int mstime =getCurrentMsTimestamp();
    long int swaptime;
    #endif
    
    #ifdef VALGRIND_GPU_DEBUG
    #else
    if (!glXMakeCurrent(display_id, child_id, glc)) { // choose this x window for manipulation
        opengllogger.log(LogLevel::normal) << "RenderGroup: render: WARNING! could not draw"<<std::endl;
    }
    #endif
    XGetWindowAttributes(display_id, child_id, &(x_window_attr));
    
    #ifdef OPENGL_TIMING
    swaptime=mstime; mstime=getCurrentMsTimestamp();
    if ( (mstime-swaptime) > 2 ) {
        std::cout << "RenderGroup: render : gxlmakecurrent timing : " << mstime-swaptime << std::endl;
    }
    #endif
    
    #ifdef PRESENT_VERBOSE
    std::cout << "RenderGroup: render: window w, h " <<x_window_attr.width<<" "<<x_window_attr.height<<std::endl;
    #endif
    
    #ifdef VALGRIND_GPU_DEBUG
    // std::cout << "VALGRIND_GPU_DEBUG" << std::endl;
    #else
    glFinish(); // TODO: debugging
    glViewport(0, 0, x_window_attr.width, x_window_attr.height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // clear the screen and the depth buffer
    #endif
    
    for(auto it=render_contexes.begin(); it!=render_contexes.end(); ++it) {
        (*it)->render(x_window_attr);
    }
    
    #ifdef OPENGL_TIMING
    swaptime=mstime; mstime=getCurrentMsTimestamp();
    if ( (mstime-swaptime) > 2 ) {
        std::cout << "RenderGroup: render : render timing         : " << mstime-swaptime << std::endl;
    }
    #endif
    
    
    if (doublebuffer_flag) {
        #ifdef PRESENT_VERBOSE
        std::cout << "RenderGroup: render: swapping buffers "<<std::endl;
        #endif
        #ifdef VALGRIND_GPU_DEBUG
        #else
        glXSwapBuffers(display_id, child_id);
        #endif
    }
    
    // glFlush();
    // glFinish();
    
    #ifdef OPENGL_TIMING
    swaptime=mstime; mstime=getCurrentMsTimestamp();
    if ( (mstime-swaptime) > 2 ) {
        std::cout << "RenderGroup: render : swap buffer timing    : " << mstime-swaptime << std::endl;
    }
    #endif
    
    // glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0); // unbind
    // glBindTexture(GL_TEXTURE_2D, 0); // unbind
    // glFinish(); // TODO: necessary?
    
    #ifdef PRESENT_VERBOSE
    std::cout << "RenderGroup: render: stop rendering!"<<std::endl;
    #endif
}



OpenGLThread::OpenGLThread(const char* name, OpenGLFrameFifoContext fifo_ctx, unsigned msbuftime, const char* x_connection) : Thread(name), infifo(new OpenGLFrameFifo(fifo_ctx)), infilter(name,infifo), msbuftime(msbuftime), x_connection(x_connection), debug(false), static_texture_file("") {
    // So, what happens here..?
    // We create the OpenGLFrameFifo instance "infifo" at constructor time, and then pass "infifo" to AVFifoFrameFilter instance "framefilter" as a parameter
    // * framefilter (AVFifoFrameFilter) knows infifo
    // * infifo (OpenGLFrameFifo) has series of stacks 
    // * The stacks are initialized by OpenGLThread at OpenGLThread::preRun()
    
    std::srand(std::time(0)); // init random number generator
    
    if (msbuftime<100) {
        opengllogger.log(LogLevel::normal) << "OpenGLThread: constructor: WARNING: your buffering time is very small! Only "<<msbuftime<< " milliseconds: lots of frames might be scrapped"<<std::endl;
    }
    
    future_ms_tolerance=msbuftime*5; // frames this much in the future will be scrapped
    
    // slot_times.resize(I_MAX_SLOTS+1,0);
    
    resetCallTime();
}


OpenGLThread::~OpenGLThread() {// WARNING: deallocations shoud be in postRun, i.e. before thread join
    stopCall();
    delete infifo;
    opengllogger.log(LogLevel::debug) << "OpenGLThread: destructor "<<std::endl;
}


unsigned OpenGLThread::getSwapInterval(GLXDrawable drawable) {
    unsigned i;
    if (drawable==0) {
        drawable=root_id;
    }
    switch (swap_flavor) {
        case swap_flavors::ext: {
            glXQueryDrawable(display_id, drawable, GLX_SWAP_INTERVAL_EXT, &i);
            break;
        }
        case swap_flavors::mesa: {
            i=(*pglXGetSwapIntervalMESA)();
            break;
        }
        default: {
            opengllogger.log(LogLevel::normal) << "OpenGLThread::getSwapInterval: could not get swap interval" << std::endl;
            break;
        }
    }
    return i;
}

void OpenGLThread::setSwapInterval(unsigned i, GLXDrawable drawable) {
    if (drawable==0) {
        drawable=root_id;
    }
    switch (swap_flavor) {
        case swap_flavors::ext: {
            // I could load this with glxext.h, but including that creates a crash
            // glXSwapIntervalEXT(display_id, drawable, i); // this is annoying .. for "GL_EXT_swap_control" we specify a drawable, for the mesa version, we don't..
            (*pglXSwapIntervalEXT)(display_id, drawable, i);
            break;
        }
        case swap_flavors::mesa: {
            (*pglXSwapIntervalMESA)(i);
            break;
        }
        default: {
            opengllogger.log(LogLevel::normal) << "OpenGLThread::setSwapInterval: could not set swap interval" << std::endl;
            break;
        }
    }
}


bool OpenGLThread::hasRenderGroup(Window window_id) {
    auto search = render_groups.find(window_id);
    
    if (search==render_groups.end()) {
        return false;
    } 
    else {
        return true;
    }
}


RenderGroup& OpenGLThread::getRenderGroup(Window window_id) {// Returns a reference to RenderGroup in OpenGLThread::render_groups
    return render_groups.at(window_id);
}


bool OpenGLThread::newRenderGroup(Window window_id) {
    if (hasRenderGroup(window_id)) {
        return false;
    }
    Window child_id;
    
    reConfigWindow(window_id);
    child_id=window_id;
    
    // child_id =getChildWindow(window_id); // X11 does not create nested windows .. that's a job for the window manager, right?
    
    RenderGroup rg(display_id, glc, window_id, child_id, doublebuffer_flag); // window_id = RenderGroup index, child_id = actual window id
    render_groups.insert(std::pair<Window,RenderGroup>(window_id,rg));
    return true;
}


bool OpenGLThread::delRenderGroup(Window window_id) {
    if (!hasRenderGroup(window_id)) {
        opengllogger.log(LogLevel::debug) << "OpenGLThread: delRenderGroup: no such group "<< window_id << std::endl;
        return false;
    }
    opengllogger.log(LogLevel::debug) << "OpenGLThread: delRenderGroup: erasing group "<< window_id << std::endl;
    
    // https://stackoverflow.com/questions/596162/can-you-remove-elements-from-a-stdlist-while-iterating-through-it
    // render_list: vector of lists, each vector element (and list) corresponds to a certain slot
    for(auto it=render_lists.begin(); it!=render_lists.end(); ++it) { // gives each list
        auto it2=it->begin(); // *(it) = list of RenderGroup(s)
        while(it2 !=it->end()) { // *(it2) = RenderGroup instance
            if ( (*it2)->getWindowId() == window_id ) {
                opengllogger.log(LogLevel::debug) << "OpenGLThread: delRenderGroup: erasing reference to "<< window_id << std::endl;
                it2=it->erase(it2);
            }
            else {
                ++it2;
            }
        }
    }
    
    render_groups.erase(render_groups.find(window_id));
    return true;
}


void OpenGLThread::newRenderContext(SlotNumber n_slot, Window window_id, int id, unsigned int z) {
    // if (!slotOk(n_slot)) {return 0;}
    if (!slotOk(n_slot)) {return;}
    
    // SlotContext& slot_context=slots_[n_slot]; // take a reference to the relevant SlotContext // debug ok
    SlotContext *slot_context=slots_[n_slot];
    
    // hasRenderGroup(window_id); // debug ok
    
    if (hasRenderGroup(window_id)) {
        RenderGroup* render_group= &render_groups.at(window_id); // reference to a RenderGroup corresponding to window_id // debug ok
        
        RenderContext* render_context = new RenderContext(slot_context, id, z);
        
        // RenderContext render_context(slot_context, id, z); // the brand new RenderContext // WARNING: this RenderContext goes out of scope and its destructor will be called
        // the instance of RenderContext created here will be deleted once you go out of scope .. and its destructor will be called, deleting its VAO, VBO, etc.
        // however, it's inserted into render_lists .. it should probably be reference counted..
        
        if (render_group->addContext(render_context)) {
            render_lists[n_slot].push_back(render_group); // render_lists[n_slot] == list of RenderGroup instances // a frame with slot number n_slot arrives => update render_group
            // n_slot => render_lists[n_slot] => a list of RenderGroups, i.e. x windows that the stream of this slot is streaming into    
            opengllogger.log(LogLevel::debug) << "OpenGLThread: newRenderContext: added new RenderContext" << std::endl;
            opengllogger.log(LogLevel::debug) << "OpenGLThread: newRenderContext: render_list at slot " << n_slot << " now with size " << render_lists[n_slot].size() << std::endl;
            // render_lists[1].size()
            // reportSlots();
            // reportRenderList();
            // return render_context.getId();
            render_contexes.insert(std::pair<int, RenderContext*>(id, render_context)); // shortcut
        }
        else {
            opengllogger.log(LogLevel::fatal) << "OpenGLThread: newRenderContext: FATAL: could not add render context " << id << std::endl;
        }
    }
    else {
        opengllogger.log(LogLevel::fatal) << "OpenGLThread: newRenderContext: FATAL: no such render group " << window_id << std::endl;
        // return 0;
    }
}


bool OpenGLThread::delRenderContext(int id) {
    bool removed=false;
    
    for(std::map<Window,RenderGroup>::iterator it=render_groups.begin(); it!=render_groups.end(); ++it) {
        RenderContext *render_context= (it->second).delContext(id);
        if (! render_context ) {
            opengllogger.log(LogLevel::debug) << "OpenGLThread: delRenderContext: no such context " << id << std::endl;
            removed=false;
        }
        else {
            opengllogger.log(LogLevel::debug) << "OpenGLThread: delRenderContext: removed context " << id << std::endl;
            render_contexes.erase(id);
            delete render_context;
            removed=true;
        }
    }
    
    // remove empty render groups from render list
    for(auto it=render_lists.begin(); it!=render_lists.end(); ++it) { // *(it) == std::list
        auto it2=it->begin(); // *(it2) == *RenderGroup
        while(it2 != it->end()) {
            if ((*it2)->isEmpty()) {// remove from list
                it2=it->erase(it2); // remove *(it2) [RenderGroup] from *(it1) [std::list]
            }
            else {
                ++it2;
            }
        }
    }
    
    return removed;
}

bool OpenGLThread::contextAddRectangle(int id, float left, float right, float top, float bottom) {
    opengllogger.log(LogLevel::debug) << "OpenGLThread: contextAddRectangle: " << id << std::endl;
    
    auto it=render_contexes.find(id);
    if (it==render_contexes.end()) {
        return false;
    }
    
    (it->second)->addRectangle(left, right, top, bottom);
    return true;
}
    
    
bool OpenGLThread::contextClearObjects(int id) {
    auto it=render_contexes.find(id);
    if (it==render_contexes.end()) {
        return false;
    }
    
    (it->second)->clearObjects();
    return true;
}
    



void OpenGLThread::delRenderContexes() {
    // for each render group, empty the render_contexes list
    for(std::map<Window,RenderGroup>::iterator it=render_groups.begin(); it!=render_groups.end(); ++it) {
        (it->second).render_contexes.erase((it->second).render_contexes.begin(),(it->second).render_contexes.end());
    }
    render_contexes.clear();
}



void OpenGLThread::loadYUVFrame(SlotNumber n_slot, YUVFrame* yuvframe){// Load PBO textures in slot n_slot
    // if (!slotOk(n_slot)) {return 0;} // assume checked (this is for internal use only)
    slots_[n_slot]->loadYUVFrame(yuvframe);
}


/* use this once we change from pointer arithmetics to std::maps
 * bool slotUsed(SlotNumber i) {
 *  if ( slots_.find(i) == slots_.end() ) {
 *    return true;
 *  } 
 *  else {
 *    return false;
 *  }
 * }
 */

bool OpenGLThread::slotUsed(SlotNumber i) {
    return slots_[i]->isUsed();
}


void OpenGLThread::activateSlot(SlotNumber i, YUVFrame* yuvframe) {
    opengllogger.log(LogLevel::crazy) << "OpenGLThread: activateSlot: "<< *yuvframe << std::endl;
    // slots_[i].activate(yuvframe->bmpars, yuv_shader);
    // slots_[i].activate(yuvframe->source_bmpars, yuv_shader);
    slots_[i]->activate(yuvframe->source_bmpars);
}


void OpenGLThread::activateSlotIf(SlotNumber i, YUVFrame* yuvframe) {
    #ifdef PRESENT_VERBOSE
    std::cout << "OpenGLThread: activateSlotIf: i=" << i << std::endl;
    #endif  
    
    if (slots_[i]->isActive()) { // we're banging on this for each reserved frame .. is this efficient?
        #ifdef NO_SLOT_REACTIVATION
        #else
        if ( // check if slot should be reactivated
            yuvframe->source_bmpars.type       ==slots_[i]->bmpars.type
            and yuvframe->source_bmpars.height ==slots_[i]->bmpars.height 
            and yuvframe->source_bmpars.width  ==slots_[i]->bmpars.width
        ) 
        { // The bitmap type of YUVFrame and YUVTEX are consistent
        }
        else {
            opengllogger.log(LogLevel::debug) << "OpenGLThread: activateSlotIf: texture dimensions changed: reactivate" << std::endl;
            activateSlot(i, yuvframe);
        }
        #endif
    } 
    else { // not activated => activate
        activateSlot(i, yuvframe);
    }
}


bool OpenGLThread::manageSlotTimer(SlotNumber i, long int mstime) {
    // return true; // WARNING: just a test
    return slots_[i]->manageTimer(mstime);
}


void OpenGLThread::render(SlotNumber n_slot) {// Render all RenderGroup(s) depending on slot n_slot
    // if (!slotOk(n_slot)) {return;} // assume checked
    // if (slots_[n_slot].isActive()) {// assume active
    for (auto it=render_lists[n_slot].begin(); it!=render_lists[n_slot].end(); ++it) { // iterator has the RenderGroup*
        (*it)->render();
    }
    //}
}


void OpenGLThread::checkSlots(long int mstime) {
    // iterate all slots
    int n_slot;
    for (n_slot=0; n_slot<slots_.size(); n_slot++) {
        if (slots_[n_slot]->isUsed()) { // is used
            #ifdef PRESENT_VERBOSE
            std::cout << "OpenGLThread: checkSlots: slot " << n_slot << std::endl;
            #endif
            slots_[n_slot]->checkIfDead(mstime);  // SlotContext puts itself in "dead" state if it has not received frames for a while ..
            #ifdef PRESENT_VERBOSE
            if (slots_[n_slot]->isDead()) {
                std::cout << "OpenGLThread: checkSlots: slot " << n_slot << " is dead" << std::endl;
            }
            #endif
            if (slots_[n_slot]->isPending(mstime)) { // If no frame received for a while, re-render
                #ifdef PRESENT_VERBOSE
                std::cout << "OpenGLThread: checkSlots: slot " << n_slot << " is pending" << std::endl;
                #endif
                render(n_slot);
            }
        } // is used
    }
}


void OpenGLThread::dumpPresFifo() {
    long int mstime=getCurrentMsTimestamp();
    long int rel_mstime;
    long int mstimestamp=9516360576679;
    
    // std::cout<<std::endl<<"OpenGLThread: dumpfifo: "<< mstime-msbuftime <<std::endl;
    for(auto it=presfifo.begin(); it!=presfifo.end(); ++it) {
        
        if (mstimestamp<((*it)->mstimestamp)) { // next smaller than previous.  Should be: [young (big value) .... old (small value)]
            std::cout<<"OpenGLThread: dumpfifo: JUMP!" << std::endl;
        }
        mstimestamp=(*it)->mstimestamp;
        
        rel_mstime=(*it)->mstimestamp-(mstime-msbuftime);
        // std::cout<<"OpenGLThread: dumpfifo: "<<**it<<" : "<< rel_mstime <<std::endl;
        // std::cout<<"OpenGLThread: dumpfifo: "<< rel_mstime <<std::endl;
        std::cout<<"OpenGLThread: dumpfifo: "<< rel_mstime << " <" << mstimestamp << "> " << std::endl;
    }
    std::cout<<"OpenGLThread: dumpfifo: "<<std::endl;
}


void OpenGLThread::diagnosis() {
    infifo->diagnosis(); // configuration etc. frames (not YUV Frames)
    std::cout << "PRESFIFO: " << presfifo.size() << std::endl;
}


void OpenGLThread::resetCallTime() {
    callswaptime =0;
    calltime     =getCurrentMsTimestamp();
    // std::cout << "OpenGLThread: resetCallTime  : " << std::endl;
}


void OpenGLThread::reportCallTime(unsigned i) {
    callswaptime =calltime;
    calltime     =getCurrentMsTimestamp();
    std::cout << "OpenGLThread: reportCallTime : ("<<i<<") "<< calltime-callswaptime << std::endl;
}


long unsigned OpenGLThread::insertFifo(Frame* f) {// sorted insert
    ///*
    // handle special (signal) frames here
    if (f->getFrameClass() == FrameClass::signal) {
        // std::cout << "OpenGLThread: insertFifo: SignalFrame" << std::endl;
        SignalFrame *signalframe = static_cast<SignalFrame*>(f);
        handleSignal(signalframe->opengl_signal_ctx);
        // recycle(f); // alias
        infifo->recycle(f);
        return 0;
    }
    else if (f->getFrameClass() == FrameClass::setup) {
        opengllogger.log(LogLevel::debug) << "OpenGLThread: insertFifo: SetupFrame: " << *f << std::endl;
        
        SetupFrame *setupframe = static_cast<SetupFrame*>(f);
        
        if (setupframe->sub_type == SetupFrameType::stream_state) { // stream state
            if (setupframe->n_slot <= I_MAX_SLOTS) { // slot ok
                SlotContext *slot_ctx = slots_[setupframe->n_slot]; // shorthand
                    
                if (setupframe->stream_state == AbstractFileState::seek) {
                    slot_ctx->loadFlag(true); // can't show present frame (after clicking seek, it's not valid anymore), must wait for a new frame
                }
                else if (setupframe->stream_state == AbstractFileState::play) {
                    slot_ctx->keepFlag(false); // don't keep on showing the last frame
                }
                else if (setupframe->stream_state == AbstractFileState::stop) {
                    slot_ctx->keepFlag(true); // it's ok to keep on showing the last frame
                }
            } // slot ok
        } // stream state
        infifo->recycle(f);
        return 0;
    }
    //*/
    // f->getFrameClass();
    
    
    /*
     *  timestamps in the presentation queue/fifo:
     *  
     *  <young                         old>
     *  90  80  70  60  50  40  30  20  10
     *  
     *  see also the comments in OpenGLThread::handleFifo()
     *  
     *  Let's take a closer look at this .. "_" designates another camera, so here
     *  we have frames from two different cameras:
     *  
     *  90_ 85 80_ 71 70_  60_ 60 51 50_ 40 40_ 31 30_ 22_ 20  15_ 10
     *  
     *  incoming 39                            |
     */
    
    bool inserted=false;
    long int rel_mstimestamp;
    
    auto it=presfifo.begin();
    while(it!=presfifo.end()) {
        #ifdef PRESENT_VERBOSE // the presentation fifo will be extremely busy.. avoid unnecessary logging
        // std::cout << "OpenGLThread: insertFifo: iter: " << **it <<std::endl;
        #endif
        if (f->mstimestamp >= (*it)->mstimestamp ) {//insert before this element
            #ifdef PRESENT_VERBOSE
            //std::cout << "OpenGLThread: insertFifo: inserting "<< *f <<std::endl; // <<" before "<< **it <<std::endl;
            #endif
            presfifo.insert(it,f);
            inserted=true;
            break;
        }
        ++it;
    }
    if (!inserted) {
        #ifdef PRESENT_VERBOSE
        // std::cout << "OpenGLThread: insertFifo: inserting "<< *f <<" at the end"<<std::endl;
        #endif
        presfifo.push_back(f);
    }
    
    rel_mstimestamp=( presfifo.back()->mstimestamp-(getCurrentMsTimestamp()-msbuftime) );
    rel_mstimestamp=std::max((long int)0,rel_mstimestamp);
    
    if (rel_mstimestamp>future_ms_tolerance) { // fifo might get filled up with frames too much in the future (typically wrong timestamps..) process them immediately
        #ifdef PRESENT_VERBOSE
        std::cout << "OpenGLThread: insertFifo: frame in distant future: "<< rel_mstimestamp <<std::endl;
        #endif
        rel_mstimestamp=0;
    }
    
    #ifdef PRESENT_VERBOSE
    std::cout << "OpenGLThread: insertFifo: returning timeout "<< rel_mstimestamp <<std::endl;
    #endif
    return (long unsigned)rel_mstimestamp; //return timeout to next frame
}


long unsigned OpenGLThread::handleFifo() {// handles the presentation fifo
    // Check out the docs for the timestamp naming conventions, etc. in \ref timing
    long unsigned mstime_delta;         // == delta
    long int      rel_mstimestamp;      // == trel = t_ - (t-tb) = t_ - delta
    Frame*        f;                    // f->mstimestamp == t_
    bool          present_frame; 
    long int      mstime;  
    
    // mstime_delta=getCurrentMsTimestamp()-msbuftime; // delta = (t-tb)
    // mstime       =getCurrentMsTimestamp();
    // mstime_delta =mstime-msbuftime;
    
    #ifdef TIMING_VERBOSE
    resetCallTime();
    #endif
    
    auto it=presfifo.rbegin(); // reverse iterator
    
    while(it!=presfifo.rend()) {// while
        // mstime_delta=getCurrentMsTimestamp()-msbuftime; // delta = (t-tb)
        mstime       =getCurrentMsTimestamp();
        mstime_delta =mstime-msbuftime;
        
        f=*it; // f==pointer to frame
        rel_mstimestamp=(f->mstimestamp-mstime_delta); // == trel = t_ - delta
        if (rel_mstimestamp>0 and rel_mstimestamp<=future_ms_tolerance) {// frames from [inf,0) are left in the fifo
            // ++it;
            break; // in fact, just break the while loop (frames are in time order)
        }
        else {// remove the frame *f from the fifo.  Either scrap or present it
            // 40 20 -20 => found -20
            ++it; // go one backwards => 20 
            it= std::list<Frame*>::reverse_iterator(presfifo.erase(it.base())); // eh.. it.base() gives the next iterator (in forward sense?).. we'll remove that .. create a new iterator on the modded
            // it.base : takes -20
            // erase   : removes -20 .. returns 20
            // .. create a new reverse iterator from 20
            present_frame=false;
            
            #ifdef NO_LATE_DROP_DEBUG // present also the late frames
            // std::cout << "OpenGLThread: rel_mstimestamp, future_ms_tolerance : " << rel_mstimestamp << " " << future_ms_tolerance << std::endl;
            if (rel_mstimestamp>future_ms_tolerance) { // fifo might get filled up with future frames .. if they're too much in the future, scrap them
                opengllogger.log(LogLevel::normal) << "OpenGLThread: handleFifo: DISCARDING a frame too far in the future " << rel_mstimestamp << " " << *f << std::endl;
            } 
            else { // .. in all other cases, just present the frame
                present_frame=true;
            }
            
            #else
            if (rel_mstimestamp<=-10) {// scrap frames from [-10,-inf)
                // opengllogger.log(LogLevel::normal) << "OpenGLThread: handleFifo: DISCARDING late frame " << " " << rel_mstimestamp << " " << *f << std::endl;
                opengllogger.log(LogLevel::normal) << "OpenGLThread: handleFifo: DISCARDING late frame " << " " << rel_mstimestamp << " <" << f->mstimestamp <<"> " << std::endl;
            }
            else if (rel_mstimestamp>future_ms_tolerance) { // fifo might get filled up with future frames .. if they're too much in the future, scrap them
                opengllogger.log(LogLevel::normal) << "OpenGLThread: handleFifo: DISCARDING a frame too far in the future " << rel_mstimestamp << " " << *f << std::endl;
            }
            else if (rel_mstimestamp<=0) {// present frames from [0,-10)
                present_frame=true;
            }
            #endif
            
            if (present_frame) { // present_frame
                if (!slotOk(f->n_slot)) {//slot overflow, do nothing
                }
                else if (f->getFrameClass()==FrameClass::yuv) {// YUV FRAME
                    YUVFrame *yuvframe = static_cast<YUVFrame*>(f);
                    #if defined(PRESENT_VERBOSE) || defined(TIMING_VERBOSE)
                    std::cout<<"OpenGLThread: handleFifo: PRESENTING " << rel_mstimestamp << " <"<< yuvframe->mstimestamp <<"> " << std::endl;
                    if (it!=presfifo.rend()) {
                        std::cout<<"OpenGLThread: handleFifo: NEXT       " << (*it)->mstimestamp-mstime_delta << " <"<< (*it)->mstimestamp <<"> " << std::endl;
                    }
                    #endif
                    // if next frame was give too fast, scrap it
                    if (manageSlotTimer(yuvframe->n_slot,mstime)) {
                        activateSlotIf(yuvframe->n_slot, yuvframe); // activate if not already active
                        #ifdef TIMING_VERBOSE
                        reportCallTime(0);
                        #endif
                        // yuv_frame's pbos have already been uploaded to GPU.  Now they're loaded to the texture
                        // loadTEX uses slots_[], where each vector element is a SlotContext (=set of textures and a shader program)
                        loadYUVFrame(yuvframe->n_slot, yuvframe);
                        #ifdef TIMING_VERBOSE
                        reportCallTime(1);
                        #endif
                        render(yuvframe->n_slot); // renders all render groups that depend on this slot.  A slot => RenderGroups (x window) => list of RenderContext => SlotContext (textures)
                        #ifdef TIMING_VERBOSE
                        reportCallTime(2);
                        #endif
                    }
                    else {
                        // opengllogger.log(LogLevel::normal) << "OpenGLThread: handleFifo: feeding frames too fast! dropping.." << std::endl; // printed by manageSlotTimer => manageTimer
                    }
                } // YUV FRAME
            }// present frame
            infifo->recycle(f); // codec found or not, frame always recycled
        }// present or scrap
    }// while
    
    if (presfifo.empty()) {
        #ifdef PRESENT_VERBOSE
        std::cout<<"OpenGLThread: handleFifo: empty! returning default " << Timeout::openglthread << " ms timeout " << std::endl;
        #endif
        return Timeout::openglthread;
    }
    else {
        f=presfifo.back();
        mstime_delta=getCurrentMsTimestamp()-msbuftime; // delta = (t-tb)
        rel_mstimestamp=f->mstimestamp-mstime_delta; // == trel = t_ - delta
        rel_mstimestamp=std::max((long int)0,rel_mstimestamp);
        #ifdef PRESENT_VERBOSE
        std::cout<<"OpenGLThread: handleFifo: next frame: " << *f <<std::endl;
        std::cout<<"OpenGLThread: handleFifo: timeout   : " << rel_mstimestamp <<std::endl;
        #endif
        return (long unsigned)rel_mstimestamp; // time delay until the next presentation event..
    }
}


void OpenGLThread::run() {// Main execution loop
    // time_t timer;
    // time_t oldtimer;
    long int mstime;
    long int old_mstime;
    
    Frame* f;
    long unsigned timeout;
    
    // time(&timer);
    // oldtimer=timer;
    mstime    =getCurrentMsTimestamp();
    old_mstime=mstime;
    
    loop=true;
    
    timeout=Timeout::openglthread;
    while(loop) {
        #ifdef PRESENT_VERBOSE
        std::cout << "OpenGLThread: "<< this->name <<" : run : timeout = " << timeout << std::endl;
        // std::cout << "OpenGLThread: "<< this->name <<" : run : dumping fifo " << std::endl;
        // infifo->dumpFifo();
        // infifo->reportStacks(); 
        // std::cout << "OpenGLThread: "<< this->name <<" : run : dumping fifo " << std::endl;
        #endif
        // std::cout << "OpenGLThread: run : read with timeout : " << timeout << std::endl;
        f=infifo->read(timeout);
        if (!f) { // TIMEOUT : either one seconds has passed, or it's about time to present the next frame..
            if (debug) {
            }
            else {
                timeout=std::min(handleFifo(),Timeout::openglthread); // present/discard frames and return timeout to the last frame.  Recycles frames.  Returns the timeout
                // std::cout << "OpenGLThread: run : no frame : timeout : " << timeout << std::endl;
            }
        } // TIMEOUT
        else { // GOT FRAME // remember to apply infifo->recycle
            if (debug) {
                opengllogger.log(LogLevel::normal) << "OpenGLThread: "<< this->name <<" : run : DEBUG MODE! recycling received frame "<< *f << std::endl;
                infifo->recycle(f);
                infifo->dumpYUVStacks();
            }
            else {
                timeout=std::min(insertFifo(f),Timeout::openglthread); // insert frame into the presentation fifo, get timeout to the last frame in the presentation fifo
                // timeout=Timeout::openglthread;
                
                // ..frame is now in the presentation fifo.. remember to recycle on handleFifo
                while (timeout==0) {
                    timeout=std::min(handleFifo(),Timeout::openglthread);
                }
                
                #if defined(PRESENT_VERBOSE) || defined(TIMING_VERBOSE)
                // dumpFifo();
                std::cout << "OpenGLThread: " << this->name <<" : run : got frame : timeout : " <<timeout<<std::endl<<std::endl;
                #endif
            }
        } // GOT FRAME
        
        // time(&timer);
        mstime=getCurrentMsTimestamp();
        
        // if (difftime(timer,oldtimer)>=1) { // time to check the signals..
        if ( (mstime-old_mstime)>=Timeout::openglthread ) {
            #ifdef PRESENT_VERBOSE
            std::cout << "OpenGLThread: " << this->name <<" : run : interrupt " << std::endl;
            #endif
            handleSignals();
            checkSlots(mstime);
            // oldtimer=timer;
            old_mstime=mstime;
            #ifdef FIFO_DIAGNOSIS
            diagnosis();
            #endif
        }
    }
}


void OpenGLThread::preRun() {// Called before entering the main execution loop, but after creating the thread
    initGLX();
    loadExtensions();
    swap_interval_at_startup=getSwapInterval(); // save the value of vsync at startup
    VSyncOff();
    makeShaders();
    dummyframe =new YUVFrame(N720);
    glFinish();
    statictex =new YUVTEX(N720);
    readStaticTex();
    
    int i;
    for(i=0;i<=I_MAX_SLOTS;i++) {
        slots_.push_back(new SlotContext(statictex,yuv_shader));
        // std::cout << ">>"<<slots_.back().active<< " " << slots_[0].active << std::endl;
    }
    for(i=0;i<=I_MAX_SLOTS;i++) {
        std::list<RenderGroup*> lis;
        render_lists.push_back(lis);
    }
    
    infifo->allocateYUV();
}


void OpenGLThread::postRun() {// Called after the main execution loop exits, but before joining the thread
    delRenderContexes();
    for(auto it=slots_.begin(); it!=slots_.end(); ++it) {
        (*it)->deActivate(); // deletes textures
        delete *it;
    }
    infifo->deallocateYUV();
    delete dummyframe;
    delShaders();
    delete statictex;
    closeGLX();
}


void OpenGLThread::handleSignal(OpenGLSignalContext &signal_ctx) {
    bool ok;
    // int i;
    
    const OpenGLSignalPars &pars = signal_ctx.pars; // alias
    
    switch (signal_ctx.signal) {
        case OpenGLSignal::exit:
            opengllogger.log(LogLevel::debug) << "OpenGLThread: handleSignals: exit" << std::endl;
            loop=false;
            break;
            
        case OpenGLSignal::info:
            reportRenderGroups();
            reportRenderList();
            break;
            
        case OpenGLSignal::new_render_group:     // newRenderCroupCall
            opengllogger.log(LogLevel::debug) << "OpenGLThread: handleSignals: new_render_group: "<< pars << std::endl;
            ok=newRenderGroup( pars.x_window_id ); // Creates a RenderGroup and inserts it into OpenGLThread::render_groups
            if (!ok) {
                opengllogger.log(LogLevel::normal) << "OpenGLThread: handleSignals: could not create render group for window id " << pars.x_window_id << std::endl;
            } 
            else {
                // pars.success=true;
            }
            break;
            
        case OpenGLSignal::del_render_group:     // delRenderGroupCall
            opengllogger.log(LogLevel::debug) << "OpenGLThread: handleSignals: del_render_group" << std::endl;
            ok=delRenderGroup( pars.x_window_id ); // Remove RenderGroup from OpenGLThread::render_groups
            if (!ok) {
                opengllogger.log(LogLevel::normal) << "OpenGLThread: handleSignals: could not delete render group for window id " << pars.x_window_id << std::endl;
            }
            else {
                // pars.success=true;
            }
            break;
            
        case OpenGLSignal::new_render_context:   // newRenderContextCall
            opengllogger.log(LogLevel::debug) << "OpenGLThread: handleSignals: new_render_context" << std::endl;
            
            // i=newRenderContext( pars->n_slot, pars->x_window_id, pars->z);
            newRenderContext( pars.n_slot, pars.x_window_id, pars.render_ctx, pars.z);
            /*
             *      if (i==0) {
             *        opengllogger.log(LogLevel::normal) << "OpenGLThread: handleSignals: could not create render context: slot, x window " << pars.n_slot << ", " << pars.x_window_id << std::endl;
    }
    // pars->render_ctx=i; // return value .. not anymore
    */
            break;
            
        case OpenGLSignal::del_render_context:   // delRenderContextCall
            opengllogger.log(LogLevel::debug) << "OpenGLThread: handleSignals: del_render_context" << std::endl;
            ok=delRenderContext(pars.render_ctx);
            if (!ok) {
                opengllogger.log(LogLevel::normal) << "OpenGLThread: handleSignals: could not delete render context " << pars.render_ctx << std::endl;
            }
            else {
                // pars->success=true;
            }
            break;
            
        case OpenGLSignal::new_rectangle:   // newRectangleCall
            opengllogger.log(LogLevel::debug) << "OpenGLThread: handleSignals: new_rectangle" << std::endl;
            ok = contextAddRectangle(pars.render_ctx, pars.object_coordinates[0], pars.object_coordinates[1], pars.object_coordinates[2], pars.object_coordinates[3]);
            if (!ok) {
                opengllogger.log(LogLevel::normal) << "OpenGLThread: handleSignals: could not add rectangle " << std::endl;
            }
            else {
                // pars->success=true;
            }
            break;
            
        case OpenGLSignal::clear_objects:   // clearObjectsCall
            opengllogger.log(LogLevel::debug) << "OpenGLThread: handleSignals: new_rectangle" << std::endl;
            ok = contextClearObjects(pars.render_ctx);
            if (!ok) {
                opengllogger.log(LogLevel::normal) << "OpenGLThread: handleSignals: could not add rectangle " << std::endl;
            }
            else {
                // pars->success=true;
            }
            break;
    }
}


void OpenGLThread::handleSignals() {
    std::unique_lock<std::mutex> lk(this->mutex);
    // AVConnectionContext connection_ctx;
    int i;
    bool ok;
    
    opengllogger.log(LogLevel::crazy) << "OpenGLThread: handleSignals: " << std::endl;
    
    // if (signal_fifo.empty()) {return;}
    
    // handle pending signals from the signals fifo
    for (auto it = signal_fifo.begin(); it != signal_fifo.end(); ++it) { // it == pointer to the actual object (struct OpenGLSignalContext)
        /* 
         * (it) == OpenGLSignalContext: members: signal, *ctx
         *    SlotNumber    n_slot;        // in: new_render_context
         *    Window        x_window_id;   // in: new_render_context, new_render_group, del_render_group
         *    unsigned int  z;             // in: new_render_context
         *    int           render_ctx;    // in: del_render_context, out: new_render_context
         */
        
        // opengllogger.log(LogLevel::debug) << "OpenGLThread: handleSignals: signal->ctx="<< *(it->ctx) << std::endl; // OpenGLSignal::exit does not need it->ctx ..
        handleSignal(*it);
    }
    
    signal_fifo.clear();
    condition.notify_one();
}


void OpenGLThread::sendSignal(OpenGLSignalContext signal_ctx) {
    std::unique_lock<std::mutex> lk(this->mutex);
    this->signal_fifo.push_back(signal_ctx);  
}


void OpenGLThread::sendSignalAndWait(OpenGLSignalContext signal_ctx) {
    std::unique_lock<std::mutex> lk(this->mutex);
    this->signal_fifo.push_back(signal_ctx);  
    while (!this->signal_fifo.empty()) {
        this->condition.wait(lk);
    }
}


void OpenGLThread::initGLX() {
    
    /*
     *  Some info about old-style/new-style context creation, multi-gpu systems, etc.
     *  
     *  https://stackoverflow.com/questions/30358693/creating-seperate-context-for-each-gpu-while-having-one-display-monitor
     *  http://on-demand.gputechconf.com/gtc/2012/presentations/S0353-Programming-Multi-GPUs-for-Scalable-Rendering.pdf
     *  https://www.khronos.org/opengl/wiki/Tutorial:_OpenGL_3.0_Context_Creation_(GLX)
     */
    
    // GLXFBConfig *fbConfigs;
    int numReturned;
    
    // initial connection to the xserver
    if (x_connection==std::string("")) {
        this->display_id = XOpenDisplay(NULL);
    }
    else {
        this->display_id = XOpenDisplay(x_connection.c_str());
    }
    if (this->display_id == NULL) {
        opengllogger.log(LogLevel::fatal) << "OpenGLThtead: initGLX: cannot connect to X server " << x_connection << std::endl;
        exit(2);
    }
    
    // glx frame buffer configuration [GLXFBConfig * list of GLX frame buffer configuration parameters] => consistent visual [XVisualInfo] parameters for the X-window
    this->root_id =DefaultRootWindow(this->display_id); // get the root window of this display
    
    /* Request a suitable framebuffer configuration - try for a double buffered configuration first */
    this->doublebuffer_flag=true;
    
    this->fbConfigs = glXChooseFBConfig(this->display_id,DefaultScreen(this->display_id),glx_attr::doubleBufferAttributes,&numReturned);
    // MEMORY LEAK when running with valgrind, see: http://stackoverflow.com/questions/10065849/memory-leak-using-glxcreatecontext
    
    this->att=glx_attr::doubleBufferAttributes;
    // this->fbConfigs = NULL; // force single buffer
    
    if (this->fbConfigs == NULL) {  /* no double buffered configs available */
        this->fbConfigs = glXChooseFBConfig( this->display_id, DefaultScreen(this->display_id),glx_attr::singleBufferAttributes,&numReturned);
        this->doublebuffer_flag=False;
        this->att=glx_attr::singleBufferAttributes;
    }
    
    if (this->fbConfigs == NULL) {
        opengllogger.log(LogLevel::fatal) << "OpenGLThread: initGLX: WARNING! no GLX framebuffer configuration" << std::endl;
    }
    
    #ifdef VALGRIND_GPU_DEBUG
    #else
    
    // get the glXCreateContextAttribsARB function from the driver .. we create here a local pointer to the function
    glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
    glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)
    glXGetProcAddress( (const GLubyte *) "glXCreateContextAttribsARB" );
    
    if (!glXCreateContextAttribsARB) {
        opengllogger.log(LogLevel::fatal) << "OpenGLThread: initGLX: FATAL! Could not create glx context: your running an old version of OpenGL"<<std::endl; 
        exit(2);
    }
    
    int context_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION, 3,
        GLX_CONTEXT_MINOR_VERSION, 0,
        //GLX_CONTEXT_FLAGS_ARB        , GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
        None
    };
    
    this->glc=glXCreateContextAttribsARB(this->display_id, this->fbConfigs[0], 0, true, context_attribs);
    
    // the old way of creating a context
    // this->glc=glXCreateNewContext(this->display_id,this->fbConfigs[0],GLX_RGBA_TYPE,NULL,True);
    
    if (!this->glc) {
        opengllogger.log(LogLevel::fatal) << "OpenGLThread: initGLX: FATAL! Could not create glx context"<<std::endl; 
        exit(2);
    }
    // this->glc=glXCreateNewContext(this->display_id,fbConfigs[0],GLX_RGBA_TYPE,NULL,False); // indirect rendering!
    // printf("ContextManager: glx context handle   =%lu\n",this->glc);
    // XFree(this->fbConfigs);
    
    // glXSwapIntervalEXT(0); // we dont have this..
    // PFNGLXSWAPINTERVALEXTPROC glXSwapIntervalEXT = (PFNGLXSWAPINTERVALEXTPROC)glXGetProcAddress((const GLubyte*)"glXSwapIntervalEXT");  // Set the glxSwapInterval to 0, ie. disable vsync!  khronos.org/opengl/wiki/Swap_Interval
    // glXSwapIntervalEXT(display_id, root_id, 0);  // glXSwapIntervalEXT(0); // not here ..
    #endif
}


void OpenGLThread::makeCurrent(Window window_id) {
    #ifdef VALGRIND_GPU_DEBUG
    #else
    glXMakeCurrent(this->display_id, window_id, this->glc);
    #endif
}


unsigned OpenGLThread::getVsyncAtStartup() {
    return swap_interval_at_startup;
}


int OpenGLThread::hasCompositor(int screen) {
    // https://stackoverflow.com/questions/33195570/detect-if-compositor-is-running
    char prop_name[20];
    snprintf(prop_name, 20, "_NET_WM_CM_S%d", screen);
    Atom prop_atom = XInternAtom(display_id, prop_name, False);
    return XGetSelectionOwner(display_id, prop_atom) != None;
}


void OpenGLThread::closeGLX() {
    XFree(this->fbConfigs);
    #ifdef VALGRIND_GPU_DEBUG
    #else
    glXDestroyContext(this->display_id, this->glc);
    #endif
    XCloseDisplay(this->display_id);
}


void OpenGLThread::loadExtensions() {
    ///*
#ifdef VALGRIND_GPU_DEBUG
#else
    this->makeCurrent(this->root_id); // a context must be made current before glew works..
    
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        // Problem: glewInit failed, something is seriously wrong.
        opengllogger.log(LogLevel::fatal) << "OpenGLThread: loadExtensions: ERROR: " << glewGetErrorString(err) <<std::endl;
        perror("Valkka: OpenGLThread: loadExtensions: ERROR:");
        exit(2);
    }
    
    /*
     *  if (GLEW_ARB_pixel_buffer_object) {
     *    opengllogger.log(LogLevel::debug) << "OpenGLThread: loadExtensions: PBO extension already loaded" <<std::endl;
     *    return;
}
else {
    opengllogger.log(LogLevel::crazy) << "OpenGLThread: loadExtensions: Will load PBO extension" <<std::endl;
}
*/
    
    /*
     *  this->makeCurrent(this->root_id); // a context must be made current before glew works..
     *  
     *  glewExperimental = GL_TRUE;
     *  GLenum err = glewInit();
     *  if (GLEW_OK != err) {
     *  // Problem: glewInit failed, something is seriously wrong.
     *  opengllogger.log(LogLevel::fatal) << "OpenGLThread: loadExtensions: ERROR: " << glewGetErrorString(err) <<std::endl;  
}
*/
    //else {
    if (GLEW_ARB_pixel_buffer_object) {
        opengllogger.log(LogLevel::debug) << "OpenGLThread: loadExtensions:  PBO extension found! :)"<<std::endl;
    }
    else {
        opengllogger.log(LogLevel::fatal) << "OpenGLThread: loadExtensions: ERROR: PBO extension not found! :("<<std::endl;
        perror("Valkka: OpenGLThread: loadExtensions: ERROR: PBO extension not found!");
        exit(2);
    }
    //}
    //*/
    
    // load some glx extensions "manually"
    ///*
    if (is_glx_extension_supported(display_id, "GLX_EXT_swap_control")) {
        std::cout << "GLX_EXT_swap_control" << std::endl;
        // typedef void ( *PFNGLXSWAPINTERVALEXTPROC) (Display *dpy, GLXDrawable drawable, int interval);
        this->pglXSwapIntervalEXT =
        (PFNGLXSWAPINTERVALEXTPROC) 
        glXGetProcAddressARB((const GLubyte *) "glXSwapIntervalEXT");
        // perror("OpenGLThread: loadExtensions: GLX_EXT_swap_control: GXL_MESA_swap_control required!");
        // swap_flavor =swap_flavors::ext;
        swap_flavor =swap_flavors::none; // lets not use this
    }
    //*/
    else if (is_glx_extension_supported(display_id, "GLX_MESA_swap_control")) {
        // std::cout << "GLX_MESA_swap_control" << std::endl;
        // typedef int (*PFNGLXGETSWAPINTERVALMESAPROC)(void);
        // typedef int (*PFNGLXSWAPINTERVALMESAPROC)(unsigned int interval);
        
        // PFNGLXGETSWAPINTERVALMESAPROC pglXGetSwapIntervalMESA =
        this->pglXGetSwapIntervalMESA =
        (PFNGLXGETSWAPINTERVALMESAPROC)
        glXGetProcAddressARB((const GLubyte *) "glXGetSwapIntervalMESA");
        
        // PFNGLXSWAPINTERVALMESAPROC pglXSwapIntervalMESA =
        this->pglXSwapIntervalMESA =
        (PFNGLXSWAPINTERVALMESAPROC)
        glXGetProcAddressARB((const GLubyte *) "glXSwapIntervalMESA"); // actually.. this seems to be in glx.h and not in glxext.h anymore..
        
        //interval = (*pglXGetSwapIntervalMESA)();
        swap_flavor =swap_flavors::mesa;
    } 
    else if (is_glx_extension_supported(display_id, "GLX_SGI_swap_control")) {
        std::cout << "GLX_SGI_swap_control" << std::endl;
        /* The default swap interval with this extension is 1.  Assume that it
         * is set to the default.
         *
         * Many Mesa-based drivers default to 0, but all of these drivers also
         * export GLX_MESA_swap_control.  In that case, this branch will never
         * be taken, and the correct result should be reported.
         */
        // interval = 1;
        
        // eh.. this is basically useless
        perror("OpenGLThread: loadExtensions: GLX_SGI_swap_control: there's something wrong with your graphics driver");
        // swap_flavor =swap_flavors::sgi;
        swap_flavor =swap_flavors::none;
    }
    else {
        perror("OpenGLThread: loadExtensions: no swap control: there's something wrong with your graphics driver");
        swap_flavor =swap_flavors::none;
    }

#endif
}


void OpenGLThread::VSyncOff() {
    setSwapInterval(0);
}


void OpenGLThread::makeShaders() {
    
    opengllogger.log(LogLevel::debug) << "OpenGLThread: makeShaders: compiling YUVShader" << std::endl;
    yuv_shader =new YUVShader();
    /*
     *  opengllogger.log(LogLevel::debug) << "OpenGLThread: makeShaders: compiling RGBShader" << std::endl;
     *  rgb_shader =new RGBShader();
     */
    if (opengllogger.log_level>=LogLevel::debug) {
        yuv_shader->validate();
        // rgb_shader->validate();
    }
}


void OpenGLThread::delShaders() {
    delete yuv_shader;
    // delete rgb_shader;
}


Window OpenGLThread::createWindow(bool map, bool show) {
    Window win_id;
    XSetWindowAttributes swa;
    
    // this->vi  =glXChooseVisual(this->display_id, 0, this->att); // "visual parameters" of the X window
    #ifdef VALGRIND_GPU_DEBUG
    // DefaultVisual(display_id, DefaultScreen(display_id));
    win_id=XCreateSimpleWindow(this->display_id, this->root_id, 10, 10, 600, 600, 2, 2, 0);
    #else
    this->vi =glXGetVisualFromFBConfig( this->display_id, this->fbConfigs[0] ); // another way to do it ..
    swa.colormap   =XCreateColormap(this->display_id, this->root_id, (this->vi)->visual, AllocNone);
    // swa.event_mask =ExposureMask | KeyPressMask;
    swa.event_mask =NoEventMask;
    // swa.event_mask =ButtonPressMask|ButtonReleaseMask;
    // swa.event_mask =ExposureMask|KeyPressMask|KeyReleaseMask|ButtonPressMask|ButtonReleaseMask|StructureNotifyMask;
    
    win_id =XCreateWindow(this->display_id, this->root_id, 0, 0, 600, 600, 0, vi->depth, InputOutput, vi->visual, CWColormap | CWEventMask, &swa);
    #endif
    
    XStoreName(this->display_id, win_id, "test window");
    if (map) {
        XMapWindow(this->display_id, win_id);
    }
    
    // win_id =glXCreateWindow(this->display_id, this->fbConfigs[0], win_id, NULL);
    
    if (show) {
        makeCurrent(win_id);
    }
    return win_id;
}


void OpenGLThread::reConfigWindow(Window window_id) {
    // glXSwapIntervalEXT(display_id, window_id, 0); // segfaults ..
    XSetWindowAttributes swa;
    // swa.colormap   =XCreateColormap(this->display_id, this->root_id, (this->vi)->visual, AllocNone);
    swa.event_mask =NoEventMask;
    // swa.event_mask =ExposureMask|KeyPressMask|KeyReleaseMask|ButtonPressMask|ButtonReleaseMask|StructureNotifyMask;
    // XChangeWindowAttributes(display_id, window_id, valuemask, attributes)
    // XChangeWindowAttributes(this->display_id, window_id, CWColormap | CWEventMask, &swa); // crashhhh
    XChangeWindowAttributes(this->display_id, window_id, CWEventMask, &swa); // ok ..
    // return glXCreateWindow(this->display_id, this->fbConfigs[0], window_id, NULL); // what crap is this..?
    // return window_id;
}


Window OpenGLThread::getChildWindow(Window parent_id) { // create new x window as a child of window_id
    // https://forum.qt.io/topic/34165/x11-reparenting-does-not-work
    Window child_id;
    
    child_id=createWindow();
    XReparentWindow(this->display_id, child_id, parent_id, 0, 0);
    // XReparentWindow(this->display_id, parent_id, child_id, 0, 0); // nopes ..
    // XMapWindow(this->display_id, child_id); 
    XFlush(this->display_id);
    return child_id;
}


void OpenGLThread::reportSlots() {
    std::cout << "OpenGLThread: reportSlots:" << std::endl;
    int i;
    
    /*
     *  for(i=0;i<int(N_MAX_SLOTS);i++) {
     *    std::cout << slots_[i].isActive() << std::endl;
     *    // std::cout << ">>"<<slots_.back().isActive()<<std::endl;
}
return;
*/
    
    i=0;
    std::cout << "OpenGLThread: reportSlots: Activated slots:" << std::endl;
    for(auto it=slots_.begin(); it!=slots_.end(); ++it) {
        if ((*it)->isActive()) {
            std::cout << "OpenGLThread: reportSlots: ACTIVE "<<i<<std::endl;
        }
        i++;
    }
    std::cout << "OpenGLThread: reportSlots:" << std::endl;
}


void OpenGLThread::reportRenderGroups() {
    std::cout << "OpenGLThread: reportRenderGroups: " << std::endl;
    std::cout << "OpenGLThread: reportRenderGroups: Render groups and their render contexes:" << std::endl;
    for(std::map<Window,RenderGroup>::iterator it=render_groups.begin(); it!=render_groups.end(); ++it) {
        std::cout << "OpenGLThread: reportRenderGroups: x window id "<< it->first << std::endl;
    }
    std::cout << "OpenGLThread: reportRenderGroups: " << std::endl;
}


void OpenGLThread::reportRenderList() {
    int i=0;
    std::cout << "OpenGLThread: reportRenderList: " << std::endl;
    std::cout << "OpenGLThread: reportRenderList: grouped render contexes" << std::endl;
    
    for(std::vector<std::list<RenderGroup*>>::iterator it=render_lists.begin(); it!=render_lists.end(); ++it) {
        if (it->size()>0) { // slot has render groups
            std::cout << "OpenGLThread: reportRenderList: groups at slot " << i <<std::endl;
            
            for(std::list<RenderGroup*>::iterator it2=it->begin(); it2!=it->end(); ++it2) {
                // std::cout << "OpenGLThread: reportRenderList:   x window id "<<(*it2)->getWindowId()<<" with "<<(*it2)->render_contexes.size() << " contexes "<<std::endl;
                std::cout << "OpenGLThread: reportRenderList:   x window id "<<(*it2)->getWindowId()<<std::endl;
                
                std::list<RenderContext*> render_contexes=(*it2)->getRenderContexes();
                
                for (std::list<RenderContext*>::iterator it3=render_contexes.begin(); it3!=render_contexes.end(); ++it3) {
                    std::cout << "OpenGLThread: reportRenderList:     * render context "<< (*it3)->getId() << std::endl;
                } // render contexes
                
            } // render groups
            
        } // if slot has render groups
        i++;
    } // slots
    std::cout << "OpenGLThread: reportRenderList: " << std::endl;
    // std::cout << "OpenGLThread: reportRenderList: " << render_lists[1].size() << std::endl;
}


// API

FifoFrameFilter &OpenGLThread::getFrameFilter() {
    return infilter;
}


void OpenGLThread::setStaticTexFile(const char* fname) {
    static_texture_file=std::string(fname);
}


void OpenGLThread::readStaticTex() {
    using std::ios;
    using std::ifstream;
    
    if (static_texture_file==std::string("")) { // no file defined
        return;
    }
    
    GLubyte*  buffer;
    int       size, ysize;
    ifstream  file;
    
    file.open(static_texture_file.c_str(),ios::in|ios::binary|ios::ate);
    
    if (file.is_open()) {
    }
    else {
        std::cout << "OpenGLThread: readStaticTex: could not open file " << static_texture_file << std::endl;
        return;
    }
    
    size = file.tellg();
    if (size>0) {
    }
    else {
        std::cout << "OpenGLThread: readStaticTex: corrupt file " << static_texture_file << std::endl;
        return;
    }
    
    buffer = new GLubyte[size];
    
    file.seekg(0,ios::beg);
    file.read((char*)buffer,size);
    
    file.close();
    
    ysize=(size*2)/3; // size of the biggest plane (Y)
    
    if (ysize!=statictex->bmpars.y_size) {
        std::cout << "OpenGLThread: readStaticTex: WARNING!  Could not read static texture from " << static_texture_file <<".  It should be 720p YUV" << std::endl;
    }
    else {
        
        /*
         *    std::cout << "size="<<size<<" ysize="<<ysize<<std::endl;
         *    int i;
         *    for(i=0;i<size;i++) {
         *      std::cout << (unsigned) buffer[i] << " ";
    }
    // return 0;
    */
        
        //Y=buffer;
        //U=buffer+ysize;
        //V=buffer+ysize+ysize/4;
        
        /*
         *    for(i=0;i<std::min(ysize,20);i++) {
         *      std::cout << U[i] << " ";
    }
    std::cout << std::endl;
    */
        
        // std::cout << "read "<<size<<" bytes"<<std::endl;
        
        statictex->loadYUV(buffer,buffer+ysize,buffer+ysize+ysize/4);
        
    }
    
    delete[] buffer;
}


void OpenGLThread::requestStopCall() {
    threadlogger.log(LogLevel::crazy) << "OpenGLThread: requestStopCall: "<< this->name <<std::endl;
    if (!this->has_thread) { return; } // thread never started
    if (stop_requested) { return; } // can be requested only once
    stop_requested = true;
    
    OpenGLSignalContext signal_ctx;
    signal_ctx.signal=OpenGLSignal::exit;
    
    threadlogger.log(LogLevel::crazy) << "OpenGLThread: sending exit signal "<< this->name <<std::endl;
    this->sendSignal(signal_ctx);
}


void OpenGLThread::infoCall() {
    OpenGLSignalContext signal_ctx;
    
    signal_ctx.signal=OpenGLSignal::info;
    sendSignal(signal_ctx);
}


bool OpenGLThread::newRenderGroupCall  (Window window_id) { // return value: render_ctx
    OpenGLSignalPars pars;
    
    pars.n_slot      =0;
    pars.x_window_id =window_id;
    pars.z           =0;
    pars.render_ctx  =0;
    pars.success     =false;
    
    opengllogger.log(LogLevel::debug) << "OpenGLThread: newRenderCroupCall: pars="<< pars <<std::endl;
    
    
    ///* // new
    SignalFrame f = SignalFrame();
    // f.opengl_signal_ctx = {OpenGLSignal::new_render_group, &pars};
    f.opengl_signal_ctx = {OpenGLSignal::new_render_group, pars};
    infilter.run(&f); // FifoFrameFilter => OpenGLFrameFifo => FrameFifo::writeCopy => |thread border|  => run : infifo.read => insertFifo (i.e. put into presentation queue (or not))
    pars.success =true;
    //*/
    
    /* // old
     *  OpenGLSignalContext signal_ctx = {OpenGLSignal::new_render_group, pars};
     *  sendSignalAndWait(signal_ctx);
     *  opengllogger.log(LogLevel::debug) << "OpenGLThread: newRenderCroupCall: return pars="<< pars <<std::endl;
     */
    
    // return pars.success;
    return true;
}


bool OpenGLThread::delRenderGroupCall  (Window window_id) {
    OpenGLSignalPars pars;
    
    pars.n_slot      =0;
    pars.x_window_id =window_id;
    pars.z           =0;
    pars.render_ctx  =0;
    pars.success     =false;
    
    // new
    SignalFrame f = SignalFrame();
    // f.opengl_signal_ctx = {OpenGLSignal::del_render_group, &pars};
    f.opengl_signal_ctx = {OpenGLSignal::del_render_group, pars};
    infilter.run(&f);
    pars.success=true;
    
    /* // old
     *  OpenGLSignalContext signal_ctx = {OpenGLSignal::del_render_group, &pars};
     *  sendSignalAndWait(signal_ctx);
     *  
     *  opengllogger.log(LogLevel::debug) << "OpenGLThread: delRenderCroupCall: return pars="<< pars <<std::endl;
     */
    
    return pars.success;
}


int OpenGLThread::newRenderContextCall(SlotNumber slot, Window window_id, unsigned int z) {
    OpenGLSignalPars pars;
    
    pars.n_slot      =slot;
    pars.x_window_id =window_id;
    pars.z           =z;
    // pars.render_ctx  =0; // return value
    pars.render_ctx  =std::rand(); // create the value here
    pars.success     =false;
    
    // /*
    // new
    SignalFrame f = SignalFrame();
    f.opengl_signal_ctx = {OpenGLSignal::new_render_context, pars};
    infilter.run(&f);
    // */
    
    /* // old
    OpenGLSignalContext signal_ctx = {OpenGLSignal::new_render_context, pars};
    sendSignalAndWait(signal_ctx);
    opengllogger.log(LogLevel::debug) << "OpenGLThread: newRenderContextCall: return pars="<< pars <<std::endl;
    */
    
    return pars.render_ctx;
}


bool OpenGLThread::delRenderContextCall(int id) {
    OpenGLSignalPars pars;
    
    pars.n_slot      =0;
    pars.x_window_id =0;
    pars.z           =0;
    pars.render_ctx  =id; // input
    pars.success     =false;
    
    ///* // new
    SignalFrame f = SignalFrame();
    // f.opengl_signal_ctx = {OpenGLSignal::del_render_context, &pars};
    f.opengl_signal_ctx = {OpenGLSignal::del_render_context, pars};
    infilter.run(&f);
    pars.success=true;
    //*/
    
    /* // old
    OpenGLSignalContext signal_ctx = {OpenGLSignal::del_render_context, pars};
    sendSignalAndWait(signal_ctx);
    opengllogger.log(LogLevel::debug) << "OpenGLThread: delRenderContextCall: return pars="<< pars <<std::endl;
    */
    return pars.success;
}

void OpenGLThread::addRectangleCall(int id, float left, float right, float top, float bottom) {
    OpenGLSignalPars pars;
    
    pars.n_slot      =0;
    pars.x_window_id =0;
    pars.z           =0;
    pars.render_ctx  =id;
    pars.success     =false;
    pars.object_coordinates = {left, right, top, bottom};
    
    SignalFrame f = SignalFrame();
    f.opengl_signal_ctx = {OpenGLSignal::new_rectangle, pars};
    infilter.run(&f);
}

void OpenGLThread::clearObjectsCall(int id) {
    OpenGLSignalPars pars;
    
    pars.n_slot      =0;
    pars.x_window_id =0;
    pars.z           =0;
    pars.render_ctx  =id;
    pars.success     =false;
    
    SignalFrame f = SignalFrame();
    f.opengl_signal_ctx = {OpenGLSignal::clear_objects, pars};
    infilter.run(&f);
}

