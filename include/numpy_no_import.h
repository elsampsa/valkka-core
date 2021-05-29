#ifndef numpy_no_import_HEADER_GUARD
#define numpy_no_import_HEADER_GUARD
/*
 * numpy_no_import.h :
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
 *  @file    numpy_no_import.h
 *  @author  Sampsa Riikonen
 *  @date    2017
 *  @version 1.2.1 
 *  
 *  @brief
 */ 

// https://github.com/numpy/numpy/issues/9309
// https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api

// #define PY_ARRAY_UNIQUE_SYMBOL valkka_shmem_array_api // this line is in common.h
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#endif
