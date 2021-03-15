#ifndef alsathread_HEADER_GUARD
#define alsathread_HEADER_GUARD
/*
 * alsathread.h :
 * 
 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
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
 *  @file    alsathread.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.1
 *  
 *  @brief
 */ 


/*
- sound frames arrive to input queue 
- 

nice references: 
https://larsimmisch.github.io/pyalsaaudio/libalsaaudio.html
https://stackoverflow.com/questions/14398573/alsa-api-how-to-play-two-wave-files-simultaneously
https://www.alsa-project.org/alsa-doc/alsa-lib/

*/

/*
struct PlaybackDeviceContext {
    std::string     name; // "default"
    int             id;   // out: identify this device
    // there can be several different ids pointing to the
    // same alsa device
    // ..that's ok for the software devices (they're mixed with dmix plugin)
};
*/

// ALSASignal::
//    new_playback_context
//          in: name, n_slot, out: playback_ctx
//    del_playback_context
//          in: playback_ctx
//  ..same things for recording contexes


//.. that should be enough.. everything else is handled internally..

#include "thread.h"
#include "framefilter.h"
#include "framefifo.h"

struct ALSAPlaybackContext {        // <pyapi>
    SlotNumber          slot;     // <pyapi>
    std::string         name;       // <pyapi>
    int                 cardindex;  // <pyapi>
    int                 id;         // <pyapi>
};                                  // <pyapi>

struct ALSARecordingContext {       // <pyapi>
    SlotNumber          slot;     // <pyapi>
    std::string         name;       // <pyapi>
    int                 cardindex;  // <pyapi>
    int                 id;         // <pyapi>
    FrameFilter&        filter;     // <pyapi>
};                                  // <pyapi>

struct ALSASignalPars {                     
    ALSAPlaybackContext     playback_ctx;  
    ALSARecordingContext    recording_ctx; 
};                                          


/** Signals for ALSAThread 
*/
enum class ALSASignal {
    none,
    exit,
    new_playback,
    new_recording,
    del_playback,
    del_recording
};

/** Encapsulate data sent to ALSAThread
*/
struct ALSASignalContext {
    ALSASignal              signal;
    ALSASignalPars          pars;
};

class ALSAThread : public Thread { // <pyapi>

public:     // <pyapi>
    ALSAThread(const char* name, FrameFifoContext fifo_ctx=FrameFifoContext()); // <pyapi>
    virtual ~ALSAThread();  // <pyapi>

protected:
    FrameFifo           *infifo;   ///< Read frames & commands from here
    FifoFrameFilter     infilter;  ///< writes to infifo

public:
    std::deque<ALSASignalContext>           signal_fifo;
    std::vector<ALSASink*>                  slots_;
    std::vector<std::vector<SetupFrame*>>   setup_frames;

public:
    void run();
    void preRun();
    void postRun();
    void sendSignal(ALSASignalContext signal_ctx);
    void requestStopCall();

private:
    void handleSignal(ALSASignalContext &signal_ctx);
    long unsigned handleFifo();
    long unsigned insertFifo(Frame* f);
    int  safeGetSlot(SlotNumber slot, FrameFilter*& ff);
    void newPlaybackSink(ALSAPlaybackContext ctx);
    void delPlaybackSink(ALSAPlaybackContext ctx);
    //void newRecordingContext(ALSARecordingContext ctx);
    //void delRecordingContext(ALSARecordingContext ctx);

public: // <pyapi>
    FifoFrameFilter &getFrameFilter();                     // <pyapi>
    void newPlaybackSinkCall(ALSAPlaybackContext ctx);       // <pyapi>
    void delPlaybackSinkCall(ALSAPlaybackContext ctx);       // <pyapi>
    //void newRecordingContextCall(ALSARecordingContext ctx);     // <pyapi>
    //void delRecordingContextCall(ALSARecordingContext ctx);     // <pyapi>
};      // <pyapi>

#endif
