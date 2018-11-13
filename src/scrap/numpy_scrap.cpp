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
