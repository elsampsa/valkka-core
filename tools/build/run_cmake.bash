#!/bin/bash
# #
# # 1) Create a separate directory for your build (let's call it $BUILD)
# # .. it doesn't have to be under the valkka main dir
# # 
# # 2) Copy this script into $BUILD
# #
# # 3) Go to $BUILD and run this script there
# #
# # (consider this: https://cmake.org/pipermail/cmake/2006-October/011711.html )
# #
# # 4) Now CMake has been configured and you can run "make" in $BUILD
# #

# # To see the meaning of these switches, run "showtags.bash" in the "src" directory
options=""

# # playing around with the debug/verbosity options ..
#
# options="-Dvalgrind_gpu_debug=OFF -Dno_late_drop_debug=OFF -Davthread_verbose=OFF -Ddecode_verbose=OFF -Dload_verbose=OFF -Dpresent_verbose=OFF -Drender_verbose=OFF -Dfifo_verbose=OFF -Dtiming_verbose=OFF -Dopengl_timing=OFF -Dfile_verbose=OFF -Dfifo_diagnosis=OFF -Dstream_send_debug=OFF"
# options="-Dfifo_diagnosis=ON"
# options="-Dfifo_diagnosis=OFF"
# options="-Dvalgrind_gpu_debug=ON -Dpresent_verbose=ON -Drender_verbose=ON"
# options="-Dpresent_verbose=ON -Dload_verbose=ON -Drender_verbose=ON"
# options="-Dfile_verbose=ON"
# options="-Dstream_send_debug=ON"
# options="-Dprofile_timing=ON"
# options="-Dpresent_verbose=ON -Dvalgrind_gpu_debug=ON"
# options="-Dpresent_verbose=ON"
# options="-Dpresent_verbose=ON -Drender_verbose=ON"

# # Choose either one of these:
build_type="Debug"
# build_type="Release"

# # live555 and ffmpeg root directories are the ones where you find their "README" files.  Use absolute paths
live555_root=$HOME"/live555/live/"
ffmpeg_root=$HOME"/ffmpeg/ffmpeg_git_lgpl/"

# # Substitute here your absolute path to the main valkka dir (where you have "CMakeLists.txt")
MY_CMAKE_DIR=/home/sampsa/C/valkka/

echo
echo $MY_CMAKE_DIR
echo
cmake $options -DCMAKE_BUILD_TYPE=$build_type -Dlive555_root=$live555_root -Dffmpeg_root=$ffmpeg_root $MY_CMAKE_DIR
echo
echo Run \"make\" or \"make VERBOSE=1\" to compile
echo Run \"make package\" to generate the .deb package
echo
