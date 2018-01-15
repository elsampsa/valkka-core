#ifndef QUEUES_HEADER_GUARD 
#define QUEUES_HEADER_GUARD
/*
 * queues.h : Lockable and safe queues for multithreading use
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
 *  @file    queues.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.2.0 
 *  
 *  @brief Lockable and safe queues for frame queueing in multithreading applications
 *
 */

#include <frames.h> 
#include <deque>
// #include <mutex>
// #include <condition_variable>


/** A thread-safe FrameFifo (first-in-first-out) queue with thread-safe methods for inserting and popping frames in/from the queue.
 * 
 * FrameFifo also implements an internal semaphore.  For this, we need three things:
 * 
 *  - an event (i.e. flag).  In cpp these are called "condition variables" (FrameFifo::condition)
 *  - a lock (FrameFifo::mutex)
 *  - a counter (FrameFifo::count)
 * 
 * Condition variable signals when there is a state change.  Lock protects inspection/manipulation of the semaphore counter.
 * 
 * Frames used in the fifo originate from an internal stack reservoir. Once the frame is removed ("popped") from the right end of the queue, it is returned ("recycled") back to the reservoir.  The process cycle is as follows:
 * 
 * - Take a Frame (let's call it "tmpframe") from the pre-allocated reservoir FrameFifo.stack
 * - Do something with Frame "tmpframe" (e.g. insert data to Frame.payload)
 * - Insert Frame into the fifo (i.e. into FrameFifo.fifo) from the left
 * - Another thread takes the frame from the right-side of FrameFifo.fifo, and does something with it (most likely creates a copy) and finally returns ("recycles") it back to FrameFifo.stack
 * 
 * These steps are implemented in FrameFifo::writeCopy and FrameFifo::recycle
 * 
 * FrameFifo::recycle also checks the payload size of the "recycled" frame (this may have been modified by the user) and augments FrameFifo.target_size accordingly.  On consecutive calls to FrameFifo::writeCopy, the frames in the stack are conformed to this payload size
 * 
 * @ingroup queues_tag
 */
class FrameFifo { // <pyapi>
 
protected:
  static void initReservoir(std::vector<Frame> &reservoir, unsigned short n_stack);
  static void initPayload(std::vector<Frame> &reservoir);
  static void initStack(std::vector<Frame> &reservoir, std::deque<Frame*> &stack);
  static void dump(std::deque<Frame*> &queue);
  
public: // <pyapi>
  /** Default constructor
   * 
   * @param n_stack   Size of the frame reservoir
   * 
   */
  FrameFifo(const char* name, unsigned short int n_stack); // <pyapi>
  /** Default virtual destructor
   */
  virtual ~FrameFifo(); // <pyapi>
  
private:
  /** Void copy-constructor: this class is non-copyable
   * 
   * We have a mutex member in this class.  Those are non-copyable.  Other possibility would be to manage a pointer to the mutex.  The "pointerization" can be done at some other level as well (say, using pointers of this object)
   */
  FrameFifo( const FrameFifo& ); //not implemented anywhere
  /** Void copy-constructor: this class is non-copyable
   */
  void operator=( const FrameFifo& ); //not implemented anywhere
  

protected: // initialized at constructor time
  std::string name;            ///< A unique name identifying this fifo
  unsigned short int n_stack;  ///< Max number of frames in the fifo
  
public:
  // insert a frame into the beginning of fifo
  // void write(Frame* f);
  
  virtual Frame* getFrame();                             ///< Take a frame from the stack
  virtual bool writeCopy(Frame* f, bool wait=false);     ///< Take a frame "ftmp" from the stack, copy contents of "f" into "ftmp" and insert "ftmp" into the beginning of the fifo (i.e. perform "copy-on-insert").  The size of "ftmp" is also checked and set to target_size, if necessary  
  // virtual bool writeCopy_(Frame* f);                     ///< Like FrameFifo::writeCopy, but waits for the fifo to have free frames
  virtual Frame* read(unsigned short int mstimeout=0);   ///< Pop a frame from the end of the fifo and return the frame to the reservoir stack
  // Frame* readCopy();                                  ///< Pop a frame from the end of the fifo, recycle it into stack and return a copy of the frame ("copy-on-read")
  virtual void recycle(Frame* f);                        ///< Return Frame f back into the stack.  Update target_size if necessary
  
  virtual void dumpStack(); ///< Dump the frames in the stack
  virtual void dumpFifo();  ///< Dump frames in the fifo
  
  // void waitAvailable();
  
protected:
  std::mutex mutex;                   ///< The Lock
  std::condition_variable condition;  ///< The Event/Flag
  unsigned long count = 0;            ///< Semaphore counter: number of frames available for the consumer
  std::vector<Frame> reservoir;       ///< Pre-allocated frames are warehoused in this reservoir
  unsigned long target_size = 0;      ///< Minimum frame payload size
  std::condition_variable ready_condition;  ///< The Event/Flag for FrameFifo::ready_mutex
  
public:
  // to use deque of objects or pointers, that's the question
  // https://stackoverflow.com/questions/10368814/what-is-the-best-way-to-clear-out-a-deque-of-pointers
  // objects : when popping / inserting, we're making shallow copies of the objects
  // pointers: .. , we'd be making copies of pointers (more faster, but really, micro-optimization)
  //
  // for various reasons, use deques of pointers
  std::deque<Frame*> stack; ///< Reservoir of frames.  We could have several reservoirs, i.e. for different frame sizes, etc.
  std::deque<Frame*> fifo;  ///< The actual fifo
  /*
  std::deque<Frame> stack; ///< reservoir of frames
  std::deque<Frame> fifo;  ///< the actual fifo
  */
 
// When object of this type goes out of scope ..
// .. Frames in the reservoir automatically d'allocated
// 
  
}; // <pyapi>


/** Passes frames to a multiprocessing fifo.
 * 
 * Typically, the terminal point for the frame filter chain, so there is no next filter = NULL.
 * 
 * @ingroup filters_tag
 * @ingroup queues_tag
 */
class FifoFrameFilter : public FrameFilter { // <pyapi>
  
public: // <pyapi>
  FifoFrameFilter(const char* name, FrameFifo& framefifo); ///< Default constructor // <pyapi>
  
protected:
  FrameFifo& framefifo;
  
protected:
  void go(Frame* frame);
}; // <pyapi>


/** Passes frames to a multiprocessing fifo.
 * 
 * Works like FifoFrameFilter, but blocks if the receiving FrameFifo does not have available frames
 * 
 * @ingroup filters_tag
 * @ingroup queues_tag
 */
class BlockingFifoFrameFilter : public FrameFilter { // <pyapi>
  
public: // <pyapi>
  BlockingFifoFrameFilter(const char* name, FrameFifo& framefifo); ///< Default constructor // <pyapi>
  
protected:
  FrameFifo& framefifo;
  
protected:
  void go(Frame* frame);
}; // <pyapi>







#endif