/*
 * valkkafs.cpp :
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
 *  @file    valkkafs.cpp
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 0.8.0 
 *  
 *  @brief 
 */ 

#include "valkkafs.h"


ValkkaFS::ValkkaFS(const char *device_file, const char *block_file, std::size_t blocksize, std::size_t n_blocks) : device_file(device_file), block_file(block_file), blocksize(blocksize), n_blocks(n_blocks), col_0(0), col_1(0), current_row(0), prev_row(0), pyfunc(NULL)
{
    tab.resize(n_cols*n_blocks, 0);
    
    device_size=blocksize*n_blocks;
 
    /*
    #define import_array() {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); } }
    import_array(); // needed here !
    
    tab2.resize(2*n_blocks, 0); // blocktable copy .. makes more sense to create the array at the main program / python side
    
    PyArray_NewFromDescr(
        PyTypeObject* subtype, 
        PyArray_Descr* descr, 
        int nd, 
        npy_intp* dims, 
        npy_intp* strides, 
        void* data, 
        int flags, 
        PyObject* obj
    )
    */
    
    /*
    npy_intp dims[] = {n_blocks, 2};
    descr = PyArray_DescrFromType(NPY_LONG); // https://stackoverflow.com/questions/42913564/numpy-c-api-using-pyarray-descr-for-array-creation-causes-segfaults
    // https://docs.scipy.org/doc/numpy-1.13.0/reference/c-api.dtype.html
    
    
    std::cout << "arr" << std::endl;
    
    arr = (PyArrayObject*)PyArray_NewFromDescr(
        &PyArray_Type,
        descr, 
        2, 
        dims, 
        NULL, 
        tab2.data(),
        NPY_ARRAY_F_CONTIGUOUS,
        NULL
    );
    
    Py_INCREF(arr);
    
    /*
    npy_intp ind[] = {0, 0};

    long *mat;
    
    mat = (long*)arr->data;
    
    mat[0] = 11;
    mat[1] = 12;
    mat[2] = 21;
    mat[3] = 22;
    // ind[0] runs first
    
    ind[0]=0; ind[1]=0;
    std::cout << "val =" << *(long*)PyArray_GetPtr(arr, ind) << std::endl;
    
    ind[0]=0; ind[1]=1;
    std::cout << "val =" << *(long*)PyArray_GetPtr(arr, ind) << std::endl;
    
    ind[0]=1; ind[1]=0;
    std::cout << "val =" << *(long*)PyArray_GetPtr(arr, ind) << std::endl;
    */
}
    
ValkkaFS::~ValkkaFS() {
    // delete py_array;
    // Py_DECREF(arr);
}

void ValkkaFS::dump() {
    std::unique_lock<std::mutex> lk(this->mutex);
    std::ofstream os(block_file, std::ios::binary);
    os.write((const char*)&tab, sizeof(tab));
    os.close();
}

void ValkkaFS::read() {
    std::unique_lock<std::mutex> lk(this->mutex);
    std::ifstream is(block_file, std::ios::binary);
    is.read((char*)&tab, sizeof(tab));
    is.close();
}
    

std::size_t ValkkaFS::ind(std::size_t i, std::size_t j) {
    if (i>=n_blocks or j>=n_cols) {
        std::cout << "ValkkaFS: ind: wrong index " << i << " " << j << std::endl;
    }
    else {
        return i*n_cols+j;
    }
}

void ValkkaFS::setVal(std::size_t i, std::size_t j, long int val) {
    std::unique_lock<std::mutex> lk(this->mutex);
    tab[ind(i,j)]=val;
}


long int ValkkaFS::getVal(std::size_t i, std::size_t j) {
    std::unique_lock<std::mutex> lk(this->mutex);
    return tab[ind(i,j)];
}

std::size_t ValkkaFS::getBlockSeek(std::size_t n_block) {
    return n_block*blocksize;
}

std::size_t ValkkaFS::getCurrentBlockSeek() {
    return getBlockSeek(current_row);
}


void ValkkaFS::writeBlock() {
    std::unique_lock<std::mutex> lk(this->mutex);
    std::string msg("");
    
    // std::cout << "current_row = " << current_row << std::endl;
    // std::cout << "prev_row = " << prev_row << std::endl;
    
    if (col_0==0) {
        std::cout << "ValkkaFS : writeBlock : no frames.  Congrats, your ValkkaFS is broken" << std::endl;
        col_0 = tab[ind(prev_row, 0)]; // copy value from previous block
        msg="frame";
    }
    if (col_1==0) {
        std::cout << "ValkkaFS : writeBlock : no keyframes.  Your ValkkaFS block size is too small" << std::endl;
        col_1 = tab[ind(prev_row, 1)]; // copy value from previous block
        msg="keyframe";
    }
    
    tab[ind(current_row, 0)] = col_0; // save values to blocktable
    tab[ind(current_row, 1)] = col_1;
    
    prev_row=current_row;
    current_row++;
    if (current_row>=n_blocks) { // wrap
        current_row=0; 
    }
    
    col_0=0;
    col_1=0;
    
    if (pyfunc!=NULL) {
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
        if (msg.size()>0) {
            PyObject_CallFunction(pyfunc, "s", msg.c_str());
        }
        else {
            PyObject_CallFunction(pyfunc, "n", current_row);
        }
        PyGILState_Release(gstate);
    }
}


void ValkkaFS::markFrame(long int mstimestamp) {
    col_1 = std::max(col_1, mstimestamp);
}
    
void ValkkaFS::markKeyFrame(long int mstimestamp) {
    col_0 = std::max(col_0, mstimestamp);
    col_1 = std::max(col_1, mstimestamp);
}


std::string ValkkaFS::getDevice() {
    return device_file;
}
    
std::size_t ValkkaFS::getDeviceSize() {
    return device_size;
}
    
void ValkkaFS::clearDevice() {
    std::unique_lock<std::mutex> lk(this->mutex);
    std::ofstream os(device_file, std::ios::binary);
    std::size_t i;

    char block[blocksize];
    memset(&block, 0, blocksize);    
    for(i=0; i<n_blocks; i++) {
        os.write((const char*)&block, blocksize);
    }
    os.close();
}

void ValkkaFS::clearTable() {
    std::unique_lock<std::mutex> lk(this->mutex);
    std::fill(tab.begin(), tab.end(), 0);
}

void ValkkaFS::setCurrentBlock(std::size_t n_block) {
    std::unique_lock<std::mutex> lk(this->mutex);
    current_row=n_block;
}


void ValkkaFS::setBlockCallback(PyObject* pobj) {
    std::unique_lock<std::mutex> lk(this->mutex);
    
    // pass here, say "signal.emit" or a function/method that accepts single argument
    if (PyCallable_Check(pobj)) { // https://docs.python.org/3/c-api/type.html#c.PyTypeObject
        Py_INCREF(pobj);
        pyfunc=pobj;
    }
    else {
        std::cout << "TestThread: setCallback: needs python callable" << std::endl;
        pyfunc=NULL;
    }
}

void ValkkaFS::setArrayCall(PyObject* pyobj) {
    std::unique_lock<std::mutex> lk(this->mutex);

    Py_INCREF(pyobj);
    
    PyArrayObject *pyarr = (PyArrayObject*)pyobj;
    long int *data = (long int*)pyarr->data;
    
    /*
    setVal(0,0,11);
    setVal(0,1,12);
    setVal(1,0,21);
    setVal(1,1,22);
    */
    /*
    tab[0]=11;
    tab[1]=12;
    tab[2]=21;
    tab[3]=22;
    */
    
    memcpy(data, tab.data(), tab.size()*sizeof(long int));
    
    /*
    data[0]=11;
    data[1]=12;
    data[2]=21;
    data[3]=22;
    */
    
    Py_DECREF(pyobj);
}



std::size_t ValkkaFS::get_n_blocks() {
    return n_blocks;    
}

std::size_t ValkkaFS::get_n_cols() {
    return n_cols;
}


ValkkaFSWriterThread::ValkkaFSWriterThread(const char *name, ValkkaFS &valkkafs, FrameFifoContext fifo_ctx) : Thread(name), valkkafs(valkkafs), infifo(name,fifo_ctx), infilter(name,&infifo), infilter_block(name,&infifo), filestream(name, std::ios::binary)
{
}
    
ValkkaFSWriterThread::~ValkkaFSWriterThread() {
    filestream.close();
}

void ValkkaFSWriterThread::run() {
    bool ok;
    unsigned short subsession_index;
    Frame* f;
    time_t timer;
    time_t oldtimer;
    long int dt;
    
    time(&timer);
    oldtimer=timer;
    loop=true;
    
    while(loop) {
        f=infifo.read(Timeout::valkkafswriterthread);
        if (!f) { // TIMEOUT
            std::cout << "ValkkaFSWriterThread: "<< this->name <<" timeout expired!" << std::endl;
        }
        else { // GOT FRAME // this must ALWAYS BE ACCOMPANIED WITH A RECYCLE CALL
            // Handle signal frames
            if (f->getFrameClass()==FrameClass::signal) {
                SignalFrame *signalframe = static_cast<SignalFrame*>(f);
                handleSignal(signalframe->valkkafswriter_signal_ctx);
            }
            else if (f->getFrameClass()==FrameClass::basic) {
                std::cout << "ValkkaFSWriterThread : " << this->name <<" got BasicFrame " << *f << std::endl;
                // TODO: use valkkafs, filestream and frame methods, declare a new block when necessary
                /*
                std::size_t calcSize();                             
                bool dump(IdNumber device_id, std::ofstream &os);
                */
            }
            else {
                std::cout << "ValkkaFSWriterThread : " << this->name <<" accepts only BasicFrame " << std::endl;
            }
            infifo.recycle(f); // always recycle
        } // GOT FRAME
        
        time(&timer);
        
        // old-style ("interrupt") signal handling
        if ( (1000*difftime(timer,oldtimer)) >= Timeout::valkkafswriterthread ) { // time to check the signals..
            handleSignals();
            oldtimer=timer;
        }
    }
}

void ValkkaFSWriterThread::preRun() {
}
    
void ValkkaFSWriterThread::postRun() {
}

void ValkkaFSWriterThread::handleSignal(ValkkaFSWriterSignalContext &signal_ctx) {
    switch (signal_ctx.signal) {
        case ValkkaFSWriterSignal::exit:
            loop=false;
            break;
        // TODO
    }
}

void ValkkaFSWriterThread::sendSignal(ValkkaFSWriterSignalContext signal_ctx) {
    std::unique_lock<std::mutex> lk(this->mutex);
    this->signal_fifo.push_back(signal_ctx);
}

void ValkkaFSWriterThread::handleSignals() {
    std::unique_lock<std::mutex> lk(this->mutex);
    // handle pending signals from the signals fifo
    for (auto it = signal_fifo.begin(); it != signal_fifo.end(); ++it) { // it == pointer to the actual object (struct SignalContext)
        handleSignal(*it);
    }
    signal_fifo.clear();
}

FifoFrameFilter &ValkkaFSWriterThread::getFrameFilter() {
    return infilter;
}

FifoFrameFilter &ValkkaFSWriterThread::getBlockingFrameFilter() {
    return (FifoFrameFilter&)infilter_block;
}

void ValkkaFSWriterThread::setSlotIdCall(SlotNumber slot, IdNumber id) { // TODO
}
 
void ValkkaFSWriterThread::seekCall(std::size_t n_block) { // TODO
}

void ValkkaFSWriterThread::requestStopCall() {
    if (!this->has_thread) { return; } // thread never started
    if (stop_requested) { return; }    // can be requested only once
    stop_requested = true;

    // use the old-style "interrupt" way of sending signals
    ValkkaFSWriterSignalContext signal_ctx;
    signal_ctx.signal = ValkkaFSWriterSignal::exit;
    
    this->sendSignal(signal_ctx);
}


    
 
ValkkaFSReaderThread::ValkkaFSReaderThread(const char *name, ValkkaFS &valkkafs) : Thread(name), valkkafs(valkkafs) {
}

ValkkaFSReaderThread::~ValkkaFSReaderThread() {
}


