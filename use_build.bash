#!/bin/bash
cd build_dir/lib
export LD_LIBRARY_PATH=$PWD
cd ../..
cd python
export PYTHONPATH=$PYTHONPATH:$PWD
cd ..
