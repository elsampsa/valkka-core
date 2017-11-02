#!/bin/bash
# # 
# # Run this script in this directory
# # (consider this: https://cmake.org/pipermail/cmake/2006-October/011711.html )
# #

# # To see the meaning of these switches, run "showtags.bash" in the "src" directory

options="-Dvalgrind_gpu_debug=OFF -Dno_late_drop_debug=OFF -Davthread_verbose=OFF -Ddecode_verbose=OFF -Dload_verbose=OFF -Dpresent_verbose=OFF -Drender_verbose=OFF -Dfifo_verbose=OFF"

# # Choose either one of these:
build_type="Debug"
# build_type="Release"

shared_opts="-Dcustom_build=ON -Duse_shared=OFF"
# # custom_build=ON    : you have compiled live555 and ffmpeg locally **RECOMMENDED**
# # custom_build=OFF   : system-wide installed header and .so files (i.e. with "apt-get install ..")
# # use_shared=ON      : dependencies of Valkka (live555 and ffmpeg) are found from other dynamic libraries 
# # use_shared=OFF     : live555 and ffmpeg are baked into Valkka as static libraries **RECOMMENDED**

# # live555 and ffmpeg root directories are the ones where you find their "README" files
live555_root=$HOME"/live555/live/"
ffmpeg_root=$HOME"/ffmpeg/ffmpeg_git_lgpl/"

lib_dirs="-Dlive555_root="$live555_root" -Dffmpeg_root="$ffmpeg_root

# # this no-brainer section creates the build directory under the valkka main directory tree
export MY_SAVE_DIR=$PWD
cd ..
export MY_CMAKE_DIR=$PWD
cd $MY_SAVE_DIR
# # .. you can also comment that out and move your build directory wherever, just define here the absolute path of your main valkka directory:
# export MY_CMAKE_DIR=/home/sampsa/C/valkka

echo
echo $MY_CMAKE_DIR
echo
cmake $options -DCMAKE_BUILD_TYPE=$build_type $shared_opts $lib_dirs $MY_CMAKE_DIR
echo
echo Run \"make\" or \"make VERBOSE=1\" to compile
echo Run \"make package\" to generate the .deb package
echo
