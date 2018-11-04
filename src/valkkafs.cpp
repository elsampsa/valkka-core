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


ValkkaFS::ValkkaFS(const char *device_file, const char *block_file, long int blocksize, long int n_blocks) : device_file(device_file), block_file(block_file), blocksize(blocksize), n_blocks(n_blocks)
{
    
    #define import_array() {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); } }
    import_array(); // needed here !
    
    npy_intp dims[] = {n_blocks, 2};
    
    /*
    std::cout << "simple" << std::endl;
    arr = PyArray_SimpleNew(2, dims, NPY_LONG);
    std::cout << "simple2" << std::endl;
    */
    
    /*
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
    
    descr = PyArray_DescrFromType(NPY_LONG); // https://stackoverflow.com/questions/42913564/numpy-c-api-using-pyarray-descr-for-array-creation-causes-segfaults
    // https://docs.scipy.org/doc/numpy-1.13.0/reference/c-api.dtype.html
    
    
    std::cout << "arr" << std::endl;
    
    arr = (PyArrayObject*)PyArray_NewFromDescr(
        &PyArray_Type,
        descr, 
        2, 
        dims, 
        NULL, 
        NULL, 
        NPY_ARRAY_C_CONTIGUOUS,
        NULL
    );
    
    Py_INCREF(arr);
    
    // int arr[] = {1,2,3,4,5};
    
    // np::ndarray *py_array = new np::ndarray;
    
    /*
    *py_array = np::from_data(arr, np::dtype::get_builtin<int>(),
                                     p::make_tuple(5),
                                     p::make_tuple(sizeof(int)),
                                     p::object());
    */
    
    std::cout << "exit ctor" << std::endl;
    
    npy_intp ind[] = {0, 0};
    
    /*
    long* val = (long*)PyArray_GetPtr(arr, ind);
    std::cout << "val =" << *val << std::endl;
    */
    
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
}
    
ValkkaFS::~ValkkaFS() {
    // delete py_array;    
}

PyObject* ValkkaFS::getNumpyArrayCall() {
    // TODO: mutex protection
    return (PyObject*)arr;
}


ValkkaFSWriterThread::ValkkaFSWriterThread(const char *name, ValkkaFS &valkkafs) : Thread(name), valkkafs(valkkafs) {
}
    
ValkkaFSWriterThread::~ValkkaFSWriterThread() {
}
    
 
ValkkaFSReaderThread::ValkkaFSReaderThread(const char *name, ValkkaFS &valkkafs) : Thread(name), valkkafs(valkkafs) {
}

ValkkaFSReaderThread::~ValkkaFSReaderThread() {
}


