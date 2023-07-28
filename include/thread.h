#ifndef THREADS_HEADER_GUARD
#define THREADS_HEADER_GUARD
/*
 * thread.h : Base class for multithreading
 * 
 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>
 *
 */

/** 
 *  @file    thread.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.5.0 
 *  
 *  @brief Base class for multithreading
 *
 */

#include "Python.h"
#include "framefifo.h"

// #define STD_THREAD 1 // keep this commented if you want to adjust processor affinity

/** An example of information context sent to the Thread inside Thread::SignalContext
* 
* Naming convention: if your Thread class is SomeThread, then name the Context to SomeThreadContext
* 
* @ingroup threading_tag
*/
struct ThreadContext
{
    int someint; // example: just an integer
};

/** List of possible signals for the thread
* 
* Naming convention: if your Thread class is SomeThread, then name the Signal to SomeSignal
* 
* @ingroup threading_tag
*/
enum class Signal
{
    none,
    exit
};

/** Encapsulates data sent by the signal
* 
* Has the enumerated signal from Signals class plus any other necessary data.
* 
* Naming convention: if your Thread class is SomeThread, then name the SignalContext to SomeSignalContext
* 
* @ingroup threading_tag
*/
struct SignalContext
{
    Signal signal;
    ThreadContext *thread_context;
};

/** A class for multithreading with a signaling system.
* 
* Thread class has a simple system for receiving signals.  Signals are placed into a mutex protected fifo queue. 
* The internal struct SignalContext defines data structures passed by the signals.  Subclasses typically implement their own SingalContext
*
* 
* @ingroup threading_tag
*/
class Thread { // <pyapi>

public: // <pyapi>
    /** Default constructor. 
  * 
  * @param name    - name of the thread
  * @param core_id - bind to a specific processor.  Default=-1, i.e. no processor affinity
  * 
  */
    Thread(const char *name); // don't include into python (this class is abstract)

    /** Destructor:**not** virtual.  Each subclass needs to invoke it's own Thread::stopCall() method
  */
    ~Thread(); // <pyapi>

private:
    /** Void copy-constructor: this class is non-copyable
   * 
   * We have a mutex member in this class.  Those are non-copyable.  Other possibility would be to manage a pointer to the mutex.  The "pointerization" can be done at some other level as well (say, using pointers of this object)
   * 
   * Copying threads is not a good idea either
   * 
   */
    Thread(const Thread &); //not implemented anywhere
    /** Void copy-constructor: this class is non-copyable
  */
    void operator=(const Thread &); //not implemented anywhere

protected:            // common variables of all Thread subclasses
    std::string name; ///< Name of the thread
    bool has_thread;  ///< true if thread has been started
    bool stop_requested;
    bool thread_joined;

    std::mutex start_mutex;                  ///< Mutex protecting start_condition
    std::condition_variable start_condition; ///< Notified when the thread has been started

    std::mutex mutex;                  ///< Mutex protecting the condition variable and signal queue
    std::condition_variable condition; ///< Condition variable for the signal queue (triggered when all signals processed).  Not necessarily used by all subclasses.

    std::mutex loop_mutex; ///< Protects thread's main execution loop (if necessary)

    std::deque<SignalContext> signal_fifo; ///< Signal queue (fifo).  Redefine in child classes.
    bool loop;                             ///< Use this boolean to control if the main loop in Thread:run should exit

protected: // threads, processor affinity, etc.
    int core_id;
#ifdef STD_THREAD
    std::thread internal_thread; ///< The actual thread instance, std::thread way
#else
    pthread_attr_t thread_attr; ///< Thread attributes, pthread_* way
    cpu_set_t cpuset;
    pthread_t internal_thread;
#endif

public: // not protected, cause we might need to test these separately
    /** Main execution loop is defined here.*/
    virtual void run() = 0;
    /** Called before entering the main execution loop, but after creating the thread */
    virtual void preRun() = 0;
    /** Called after the main execution loop exits, but before joining the thread */
    virtual void postRun() = 0;
    /** Called before the thread is joined **/
    virtual void preJoin();
    /** Called after the thread has been joined **/
    virtual void postJoin();
    /** Send a signal to the thread */
    virtual void sendSignal(SignalContext signal_ctx);
    /** Send a signal to the thread and wait for all signals to be executed */
    virtual void sendSignalAndWait(SignalContext signal_ctx);

protected:
    void mainRun();     ///< Does the preRun, run, postRun sequence
    void closeThread(); ///< Sends exit signal to the thread, calls join.  This method blocks until thread has exited.  Set Thread::has_thread to false.
#ifdef STD_THREAD
#else
    static void *mainRun_(void *p);
#endif

public: // *** API ***                                                  // <pyapi>
    /** API method for setting the thread affinity.  Use before starting the thread */
    void setAffinity(int i); // <pyapi>

    /** API method: starts the thread */
    void startCall(); // <pyapi>

    /** API method: stops the thread.
   * If Thread::has_thread is true, sends exit signal to the thread and calls Thread::closeThread
   * Waits until the thread is joined
   */
    virtual void stopCall(); // <pyapi>

    /** API method: stops the thread.  Like Thread::stopCall() but does not block.
   * Waiting for the thread to join is done in Thread::waitStoppedCall()
   */
    virtual void requestStopCall(); // <pyapi>

    /** API method: waits until the thread is joined.  Use with Thread::requestStopCall
   */
    virtual void waitStopCall(); // <pyapi>

    /** Wait until thread has processed all its signals */
    virtual void waitReady(); // <pyapi>

}; // <pyapi>

/**
  * @brief A demo thread for testing the producer/consumer module for fifos.  Producer side.
  * 
  * @ingroup threading_tag
  */
class TestProducerThread : public Thread
{

public:
    TestProducerThread(const char *name, FrameFifo *framefifo, int index = 0);

public:
    void run();
    void preRun();
    void postRun();

protected:
    FrameFifo *framefifo; ///< Feed frames here

private:
    int index;
};

/**
  * @brief A demo thread for testing the producer/consumer module for fifos.  Consumer side.
  * 
  * @ingroup threading_tag
  * 
  */
class TestConsumerThread : public Thread
{

public:
    TestConsumerThread(const char *name, FrameFifo *framefifo);

public:
    void run();
    void preRun();
    void postRun();

protected:
    FrameFifo *framefifo; ///< Consume frames from here
};

#endif
