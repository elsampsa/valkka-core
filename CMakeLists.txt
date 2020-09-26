# Cmake file for Valkka library
# (C) 2017, 2018 Sampsa Riikonen
#
# cmake .
# make
# (or make VERBOSE=1)
#

cmake_minimum_required(VERSION 3.1)
find_package(PkgConfig REQUIRED)
# find_package(PythonLibs 2.7 REQUIRED)

project(Valkka)

# **** INPUT VARIABLES ****
option(ffmpeg_root  "ffmpeg_root"  "")
option(live555_root "live555_root" "")

SET(FFMPEG_ROOT ${ffmpeg_root})
SET(LIVE555_ROOT ${live555_root})

# debug/verbosity switches: generate with "tools/cmake/gen_opts.py"
option(valgrind_gpu_debug "valgrind_gpu_debug" OFF)
if    (valgrind_gpu_debug)
  add_definitions("-DVALGRIND_GPU_DEBUG")
  message("VALGRIND_GPU_DEBUG ENABLED")
endif (valgrind_gpu_debug)

option(no_late_drop_debug "no_late_drop_debug" OFF)
if    (no_late_drop_debug)
  add_definitions("-DNO_LATE_DROP_DEBUG")
  message("NO_LATE_DROP_DEBUG ENABLED")
endif (no_late_drop_debug)

option(avthread_verbose "avthread_verbose" OFF)
if    (avthread_verbose)
  add_definitions("-DAVTHREAD_VERBOSE")
  message("AVTHREAD_VERBOSE ENABLED")
endif (avthread_verbose)

option(decode_verbose "decode_verbose" OFF)
if    (decode_verbose)
  add_definitions("-DDECODE_VERBOSE")
  message("DECODE_VERBOSE ENABLED")
endif (decode_verbose)

option(load_verbose "load_verbose" OFF)
if    (load_verbose)
  add_definitions("-DLOAD_VERBOSE")
  message("LOAD_VERBOSE ENABLED")
endif (load_verbose)

option(present_verbose "present_verbose" OFF)
if    (present_verbose)
  add_definitions("-DPRESENT_VERBOSE")
  message("PRESENT_VERBOSE ENABLED")
endif (present_verbose)

option(render_verbose "render_verbose" OFF)
if    (render_verbose)
  add_definitions("-DRENDER_VERBOSE")
  message("RENDER_VERBOSE ENABLED")
endif (render_verbose)

option(fifo_verbose "fifo_verbose" OFF)
if    (fifo_verbose)
  add_definitions("-DFIFO_VERBOSE")
  message("FIFO_VERBOSE ENABLED")
endif (fifo_verbose)

option(timing_verbose "timing_verbose" OFF)
if    (timing_verbose)
  add_definitions("-DTIMING_VERBOSE")
  message("TIMING_VERBOSE ENABLED")
endif (timing_verbose)

option(opengl_timing "opengl_timing" OFF)
if    (opengl_timing)
  add_definitions("-DOPENGL_TIMING")
  message("OPENGL_TIMING ENABLED")
endif (opengl_timing)

option(decode_timing "decode_timing" OFF)
if    (decode_timing)
  add_definitions("-DDECODE_TIMING")
  message("DECODE_TIMING ENABLED")
endif (decode_timing)

option(file_verbose "file_verbose" OFF)
if    (file_verbose)
  add_definitions("-DFILE_VERBOSE")
  message("FILE_VERBOSE ENABLED")
endif (file_verbose)

option(fifo_diagnosis "fifo_diagnosis" OFF)
if    (fifo_diagnosis)
  add_definitions("-DFIFO_DIAGNOSIS")
  message("FIFO DIAGNOSIS ENABLED")
endif (fifo_diagnosis)

option(stream_send_debug "stream_send_debug" OFF)
if    (stream_send_debug)
  add_definitions("-DSTREAM_SEND_DEBUG")
  message("STREAM SEND DEBUG ENABLED")
endif (stream_send_debug)

option(profile_timing "profile_timing" OFF)
if    (profile_timing)
  add_definitions("-DPROFILE_TIMING")
  message("PROFILE TIMING ENABLED")
endif (profile_timing)

option(cachestream_verbose "cachestream_verbose" OFF)
if    (cachestream_verbose)
  add_definitions("-DCACHESTREAM_VERBOSE")
  message("CACHESTREAM_VERBOSE ENABLED")
endif (cachestream_verbose)


# ****************************************************************
# WARNING: the following three lines are modified by the "setver.bash" script
SET(MAJOR_VERSION "1") 
SET(MINOR_VERSION "0") 
SET(PATCH_VERSION "0") 
set(VERSION_STRING ${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION})

# *** define build type: Debug or Release # now from the command line
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

# ****************************************************************

# [some directory definitions]
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# [OpenGL dependencies]
# exec_program(pkg-config ARGS --cflags x11 glew OUTPUT_VARIABLE GL_CFLAGS)
# exec_program(pkg-config ARGS --libs x11 glew OUTPUT_VARIABLE GL_LIBS)
pkg_search_module(X11 REQUIRED x11)
pkg_search_module(GLEW REQUIRED glew)
pkg_search_module(ALSA REQUIRED alsa) # can't remove alsa dependency from the latest ffmpeg
# pkg_search_module(PYTHON REQUIRED python3) # don't use! https://bugs.python.org/issue36721

# some cmake bullshit for a change: get python libs and cflags
exec_program(python3-config ARGS --includes OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS_TMP)
exec_program(python3-config ARGS --libs --embed OUTPUT_VARIABLE PYTHON_LIBRARIES_TMP RETURN_VALUE RETVAL)
#message("RETVAL: ${RETVAL}")
if(RETVAL)
    message("python3-config failed with --embed, so will re-run without it")
    exec_program(python3-config ARGS --libs OUTPUT_VARIABLE PYTHON_LIBRARIES_TMP RETURN_VALUE RETVAL)
else(RETVAL)
    # nada
endif(RETVAL)


string(REPLACE "-l" " " PYTHON_LIBRARIES_TMP ${PYTHON_LIBRARIES_TMP})
string(REPLACE " " ";" PYTHON_LIBRARIES ${PYTHON_LIBRARIES_TMP})
string(REPLACE "-I" " " PYTHON_INCLUDE_DIRS_TMP ${PYTHON_INCLUDE_DIRS_TMP})
string(REPLACE " " ";" PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS_TMP})

pkg_search_module(OPENSSL REQUIRED openssl)

message("PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
message("PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")

message("*** PYTHON INTERFACE ***")
# execute_process(COMMAND python3 -c "from distutils import sysconfig; print(sysconfig.get_python_lib(),end='')" OUTPUT_VARIABLE PYTHON_DIR)
set(PYTHON_DIR "lib/python3/dist-packages") # just hard-code it
message("PYTHON INSTALL DIR           : " ${PYTHON_DIR})

execute_process(COMMAND python3 -c "import numpy; print(numpy.get_include(), end='')" OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)
execute_process(COMMAND python3 -c "import numpy; print(numpy.__version__, end='')" OUTPUT_VARIABLE NUMPY_VERSION)

message("PYTHON NUMPY HEADER FILES IN : " ${NUMPY_INCLUDE_DIR})

add_definitions(-DNUMPY_VERSION="${NUMPY_VERSION}")

# pip3 install: (1.14.5)
# /home/sampsa/.local/lib/python3.5/site-packages/numpy/__init__.py
# pip3 install as sudo:
# /usr/local/lib/python3.5/dist-packages/numpy/__init__.py
# apt-get install python3-numpy: (1.11)
# /usr/lib/python3/dist-packages/numpy/__init__.py
#
# pip3 install pyqt5 imutils
# 

execute_process(
  COMMAND ./makeswig.bash
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/include
)

execute_process(
  COMMAND swig -python -c++ -I/usr/include -o ${CMAKE_SOURCE_DIR}/python/valkka/core/valkka_core_wrap.cpp -outdir ${CMAKE_SOURCE_DIR}/python/valkka/core ${CMAKE_SOURCE_DIR}/include/valkka_core.i
)
# /usr/include needed here : in valkka_core.i in the %inline section we refer to that header file


# compiler flags
# add_compile_options("-std=c++14" "-pthread") #  ${GL_CFLAGS})
add_compile_options("-std=c++14") # .. -pthread required only at link time

# [define library source files]
file(GLOB SOURCES src/*.cpp)
file(GLOB SWIGBASE include/valkka.i.base)

# [we're compiling a library here..]
add_library(${PROJECT_NAME} SHARED ${SOURCES})
# => not the target_* commands work

# [define library header files]
# include_directories(include)
target_include_directories(${PROJECT_NAME} PUBLIC include)

# [set shared library version]
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${VERSION_STRING} SOVERSION ${MAJOR_VERSION})

# [add dependencies]
target_include_directories(${PROJECT_NAME} PUBLIC ${X11_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
# # target_link_libraries(${PROJECT_NAME} ${GL_LIBS})
target_link_libraries(${PROJECT_NAME} ${X11_LIBRARIES} ${GLEW_LIBRARIES} ${ALSA_LIBRARIES} ${PYTHON_LIBRARIES})
target_link_libraries(${PROJECT_NAME} "pthread")

# [add openssl dependencies]
target_link_libraries(${PROJECT_NAME} ${OPENSSL_LIBRARIES})

message("LIVE555 ROOT: ${LIVE555_ROOT}")
message("FFMPEG  ROOT: ${FFMPEG_ROOT}")

target_include_directories(${PROJECT_NAME} PUBLIC "${LIVE555_ROOT}/liveMedia/include" "${LIVE555_ROOT}/groupsock/include" "${LIVE555_ROOT}/BasicUsageEnvironment/include" "${LIVE555_ROOT}/UsageEnvironment/include")
target_include_directories(${PROJECT_NAME} PUBLIC "${FFMPEG_ROOT}") # ffmpeg header files are referred in the code with the directory name
  
target_link_libraries(${PROJECT_NAME} "-L${LIVE555_ROOT}/liveMedia" "-L${LIVE555_ROOT}/groupsock" "-L${LIVE555_ROOT}/BasicUsageEnvironment" "-L${LIVE555_ROOT}/UsageEnvironment")
target_link_libraries(${PROJECT_NAME} "-L${FFMPEG_ROOT}/libavfilter" "-L${FFMPEG_ROOT}/libavformat" "-L${FFMPEG_ROOT}/libavcodec" "-L${FFMPEG_ROOT}/libavutil" "-L${FFMPEG_ROOT}/libswscale" "-L${FFMPEG_ROOT}/libswresample")
    
# target_link_libraries(${PROJECT_NAME} "vda") # could this fix the missing "vaUnmapBuffer", etc. crap?
target_link_libraries(${PROJECT_NAME} "rt")
target_link_libraries(${PROJECT_NAME} "-Wl,--allow-multiple-definition" "-Wl,-Bsymbolic" "-Wl,--start-group" "-Wl,--whole-archive")

target_link_libraries(${PROJECT_NAME} ":libliveMedia.a" ":libgroupsock.a" ":libBasicUsageEnvironment.a" ":libUsageEnvironment.a")
target_link_libraries(${PROJECT_NAME} ":libavfilter.a" ":libavformat.a" ":libavcodec.a" ":libavutil.a" ":libswscale.a" ":libswresample.a")

target_link_libraries(${PROJECT_NAME} "-Wl,--no-whole-archive" "-Wl,--end-group")
    
# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND "objcopy" ARGS "-N" "vaUnmapBuffer" $<TARGET_FILE:${PROJECT_NAME}>)

# WARNING: this is code generation..!  No sense to do it here.  All code should be generated at the cmake configuration stage .. look for the execute_process commands up there
#
# *** (SWIG, 1-3) SWIG PYTHON INTERFACE ***
#
#
#
# *** (SWIG, 1) Generate SWIG interface file from the header files ***
#add_custom_target(
#  swig_file ALL
#  COMMAND ./makeswig.bash
#  COMMENT GENERATING_SWIG 
#  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/include
#)
#
# *** (SWIG, 2) Generate cpp wrappers for python using SWIG ***
#add_custom_command(
#  TARGET swig_file
#  POST_BUILD
#  COMMENT SWIGGING
#  COMMAND swig -python -c++ -I/usr/include -o ${CMAKE_SOURCE_DIR}/python/valkka/valkka_core_wrap.cpp -outdir ${CMAKE_SOURCE_DIR}/python/valkka ${CMAKE_SOURCE_DIR}/include/valkka_core.i
#)

# *** (SWIG, 3) Compile the cpp-wrapped python code ***
add_library(valkka_core SHARED ${CMAKE_SOURCE_DIR}/python/valkka/core/valkka_core_wrap.cpp)
target_include_directories(valkka_core PUBLIC include)
target_include_directories(valkka_core PUBLIC ${X11_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_include_directories(valkka_core PUBLIC "${LIVE555_ROOT}/liveMedia/include" "${LIVE555_ROOT}/groupsock/include" "${LIVE555_ROOT}/BasicUsageEnvironment/include" "${LIVE555_ROOT}/UsageEnvironment/include")
target_include_directories(valkka_core PUBLIC "${FFMPEG_ROOT}") # ffmpeg header files are referred in the code with the directory name
target_include_directories(valkka_core PUBLIC "${NUMPY_INCLUDE_DIR}") # ffmpeg header files are referred in the code with the directory name

target_link_libraries(valkka_core "Valkka.so")
target_link_libraries(valkka_core "-L${CMAKE_CURRENT_BINARY_DIR}/lib")

set_target_properties(valkka_core PROPERTIES VERSION ${VERSION_STRING} SOVERSION ${MAJOR_VERSION})
set_target_properties(valkka_core PROPERTIES PREFIX "")
set_target_properties(valkka_core PROPERTIES OUTPUT_NAME "_valkka_core")
set_target_properties(valkka_core PROPERTIES SUFFIX ".so")
set_target_properties(valkka_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/valkka/core)

add_dependencies(valkka_core ${PROJECT_NAME}) # swig .so depends on the main shared library

# # original commands

# # compile:
# x86_64-linux-gnu-gcc -pthread -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/home/sampsa/.local/lib/python3.5/site-packages/numpy/core/include -I/usr/include/python3.5m -c valkka_core_wrap.cpp -o build/temp.linux-x86_64-3.5/valkka_core_wrap.o -fPIC -std=c++14

# # link:
# x86_64-linux-gnu-g++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 build/temp.linux-x86_64-3.5/valkka_core_wrap.o -L/home/sampsa/C/valkka/build_dir/lib -lValkka -o build/lib.linux-x86_64-3.5/_valkka_core.cpython-35m-x86_64-linux-gnu.so

# *** test programs ***
# use the script "test/list.bash" to update this list

# set(TESTNAMES "av_live_thread_test" "av_thread_test" "classes_test" "fifo_test" "file_test" "frames_test" "live_stream_check_test" "live_streaming_test" "live_thread_test" "log_test" "opengl_test" "opengl_thread_test" "render_test" "shmem_test" "swscale_test" "threads_test")
# set(TESTNAMES "shmem_test" "swscale_test")
# set(TESTNAMES "file_test")
set(TESTNAMES "frame_test" "framefifo_test" "thread_test" "livethread_test" 
"avthread_test" "av_live_thread_test" "openglframefifo_test" "openglthread_test" 
"live_av_openglthread_test" "live_av_openglthread_test2" "live_av_shmem_test" "file_test" 
"filethread_test" "filethread_test0" "livethread_rtsp_test" "switch_test" "framefilter_test" 
"cache_test" "usbthread_test" "valkkafs_test" "valkkafswriter_test" "cachestream_test" 
"cachestream_decode_test" "rawrite_test" "movement_test" "rgbframefifo_test" "ringbuffer_test" 
"fdwritethread_test" "mux_test" "shmem_test" "live_muxshmem_test")

add_custom_target(tests) # Note: without 'ALL'
foreach( testname ${TESTNAMES} )
  add_executable(${testname} "test/${testname}.cpp") # Note: without 'ALL'
  target_include_directories(${testname} PUBLIC "include")
  target_include_directories(${testname} PUBLIC "${LIVE555_ROOT}/liveMedia/include" "${LIVE555_ROOT}/groupsock/include" "${LIVE555_ROOT}/BasicUsageEnvironment/include" "${LIVE555_ROOT}/UsageEnvironment/include")
  target_include_directories(${testname} PUBLIC "${FFMPEG_ROOT}")

  target_include_directories(${testname} PUBLIC ${PYTHON_INCLUDE_DIRS})
  target_link_libraries(${testname} ${PYTHON_LIBRARIES})
  
  target_link_libraries(${testname} "Valkka.so")
  target_link_libraries(${testname} "-L${CMAKE_CURRENT_BINARY_DIR}/lib")
  # target_link_libraries(${testname} "-Wl,--unresolved-symbols=ignore-all") # even when all external libraries from ffmpeg have been dropped some stupid symbols persist, "vaUnmapBuffer", etc.
  # .. but then it will bust when using the python bindings
  # .. all that crap comes from libavdevice
  
  target_link_libraries(${testname} ${OPENSSL_LIBRARIES})

  add_dependencies(tests ${testname}) # tests depends on the executable
  add_dependencies(${testname} ${PROJECT_NAME}) # test depends on libValkka
endforeach( testname ${TESTNAMES} )

# *** packaging ***
# SET(CPACK_SET_DESTDIR "on") # don't use
# SET(CPACK_PACKAGING_INSTALL_PREFIX "/tmp") # don't use

exec_program(dpkg ARGS --print-architecture OUTPUT_VARIABLE MY_ARCH)

SET(CPACK_GENERATOR "DEB")
SET(CPACK_PACKAGE_NAME ${PROJECT_NAME})

SET(CPACK_PACKAGE_VERSION_MAJOR "${MAJOR_VERSION}")
SET(CPACK_PACKAGE_VERSION_MINOR "${MINOR_VERSION}")
SET(CPACK_PACKAGE_VERSION_PATCH "${PATCH_VERSION}")
SET(CPACK_PACKAGE_VERSION "${VERSION_STRING}")

SET(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${MY_ARCH})

# The dependencies: keep these consistent with debian/control
# SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libpython3.5(>= 3.5.2), libgcc1(>= 1:6.0.1), libc6(>= 2.23), libgl1-mesa-glx(>= 12.0.6), libx11-6(>= 2:1.6.3), libstdc++6(>= 5.4.0), libc6(>= 2.23), libglew1.13(>= 1.13.0), python3-numpy") # ubuntu 16
# NEW: Avoid version numbers, use "utils" packages to imply a dependency, for example, to libglew (without hardcoding the version number into the package name)
SET(CPACK_DEBIAN_PACKAGE_DEPENDS "python3, mesa-utils, libstdc++5, glew-utils, python3-numpy, v4l-utils, python3-pip")

#
# objdump -p libValkka.so
# dpkg -S libc.so.6
# => libc6:amd64: /lib/x86_64-linux-gnu/libc.so.6
# apt-cache show libc6 | grep "Version"
# => Version: 2.23-0ubuntu9
# a typical dependency seems to be: libc6 (>= 2.17)

SET(CPACK_PACKAGE_CONTACT "sampsa.riikonen@iki.fi")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Open source video management for linux")
SET(CPACK_DEBIAN_PACKAGE_RECOMMENDS "")
SET(CPACK_DEBIAN_PACKAGE_SUGGESTS "")
# SET(CPACK_PACKAGE_INSTALL_DIRECTORY "dir") # don't use

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib) # install the shared library
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include" DESTINATION include/valkka FILES_MATCHING PATTERN "*.h") # install header files
# The install command: https://cmake.org/cmake/help/v3.0/command/install.html

# Cmake's INSTALL command is totally cryptic
# what the INSTALL command (maybe) does ..
# .. it takes the last bit of DIRECTORY and puts matched files into DESTINATION/last_bit

# Cmake manual:
# "The last component of each directory name is appended to the destination directory but a trailing slash may be used to avoid this because it leaves the last component empty."
# fair enough! :)

# include header files ..
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/" DESTINATION include/valkka FILES_MATCHING PATTERN "*.h")

# install(DIRECTORY "${LIVE555_ROOT}/liveMedia/include" "${LIVE555_ROOT}/groupsock/include" "${LIVE555_ROOT}/BasicUsageEnvironment/include" "${LIVE555_ROOT}/UsageEnvironment/include" DESTINATION include/valkka FILES_MATCHING PATTERN "*.hh")

install(DIRECTORY "${LIVE555_ROOT}/liveMedia/include"               DESTINATION include/valkka/liveMedia             FILES_MATCHING PATTERN "*.h*")
install(DIRECTORY "${LIVE555_ROOT}/groupsock/include"               DESTINATION include/valkka/groupsock             FILES_MATCHING PATTERN "*.h*")
install(DIRECTORY "${LIVE555_ROOT}/BasicUsageEnvironment/include"   DESTINATION include/valkka/BasicUsageEnvironment FILES_MATCHING PATTERN "*.h*")
install(DIRECTORY "${LIVE555_ROOT}/UsageEnvironment/include"        DESTINATION include/valkka/UsageEnvironment      FILES_MATCHING PATTERN "*.h*")

# install(DIRECTORY "${LIVE555_ROOT}"        DESTINATION include/valkka FILES_MATCHING PATTERN "*.h*") # does the right thing .. why? ****ing CMake

# include header files if someone want's to use the cpp API.  Must take care thought, that the API user uses the correct header files (i.e. these, instead of some other versions)
install(DIRECTORY "${FFMPEG_ROOT}/libavfilter"   DESTINATION include/valkka   FILES_MATCHING PATTERN "*.h") # don't really need this ..
install(DIRECTORY "${FFMPEG_ROOT}/libavformat"   DESTINATION include/valkka   FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "${FFMPEG_ROOT}/libavcodec"    DESTINATION include/valkka   FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "${FFMPEG_ROOT}/libavutil"     DESTINATION include/valkka   FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "${FFMPEG_ROOT}/libswscale"    DESTINATION include/valkka   FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "${FFMPEG_ROOT}/libswresample" DESTINATION include/valkka   FILES_MATCHING PATTERN "*.h")

# TODO: change from absolute to relative dir
install(DIRECTORY "${CMAKE_SOURCE_DIR}/python/valkka"  DESTINATION ${PYTHON_DIR}  FILES_MATCHING PATTERN "*.py")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/python/valkka"  DESTINATION ${PYTHON_DIR}  FILES_MATCHING PATTERN "*.so*")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/python/valkka"  DESTINATION ${PYTHON_DIR}  FILES_MATCHING PATTERN "*.png")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/python/valkka"  DESTINATION ${PYTHON_DIR}  FILES_MATCHING PATTERN "*.yuv")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/python/valkka/onvif/wsdl"  DESTINATION "${PYTHON_DIR}/valkka/onvif"  FILES_MATCHING PATTERN "*")

# install(DIRECTORY "${FFMPEG_ROOT}" DESTINATION include/valkka   FILES_MATCHING PATTERN "*.h")

# when compiling on my linux box, I use:
# -I/home/sampsa/live555/live/UsageEnvironment/include
# -I/home/sampsa/ffmpeg/ffmpeg_git_lgpl 
#
# with cpp api, should use
# -I/usr/valkka/include/
# -I/usr/valkka/BasicUsageEnvironment/include/
# .. etc. for live555
# -I/usr/valkka/ffmpeg/
#
# however, if combined with "-I/usr/include", there might be other versions of the same header files in "/usr/include/" TODO: think about this..
#

# TODO: how to configure pkg-config when installing this .deb package?
# something like this..?
# https://gitlab.kitware.com/third-party/zlib/commit/ca6e7a0d552e3b54c0833658409e34f9de3bead6

# This must always be last!
INCLUDE(CPack)

# readings on cpack
# https://cmake.org/Wiki/CMake:CPackPackageGenerators#DEB_.28UNIX_only.29
# https://cmake.org/Wiki/CMake:Packaging_With_CPack
# https://nowardev.wordpress.com/2012/05/16/create-debian-package-for-script-and-simple-project-with-cmake-and-cpack/
# https://cmake.org/pipermail/cmake/2012-January/048781.html

#if(MAKE_TEST_BINARIES)
#  # [make cmake aware of the test binaries]
#  foreach( testname ${TESTNAMES} )
#    # message(${testname})
#    add_executable( ${testname} "test/${testname}.cpp" )
#    target_link_libraries( ${testname} ${PROJECT_NAME} )
#  endforeach( testname ${TESTNAMES} )
#endif(MAKE_TEST_BINARIES)


# [some fooling around .. keep commented]
#
# *** trying out cmake .. testing wtf is wrong with add_executable command *** [1]
# add_executable(fifo_test tests/fifo_test.cpp) # ok
# add_executable(bin/fifo_test tests/fifo_test.cpp) # does not work! add_executable does not accept paths..!
#
# *** Generate test executables, as per https://stackoverflow.com/questions/14306642/adding-multiple-executables-in-cmake ***
#file( GLOB APP_SOURCES RELATIVE ${CMAKE_CURRENT_BINARY_DIR} tests/*.cpp )
#foreach( testsourcefile ${APP_SOURCES} )
#    string( REPLACE ".cpp" "" testname ${testsourcefile} )
#    message(${testname} " " ${testsourcefile})
#    add_executable( ${testname} ${testsourcefile} )
#    target_link_libraries( ${testname} YourLib )
# endforeach( testsourcefile ${APP_SOURCES} )
#
# useful links:
# https://cmake.org/cmake/help/latest/manual/cmake-commands.7.html
# https://cmake.org/Wiki/CMake/Language_Syntax#CMake_supports_boolean_variables.
# https://cmake.org/cmake/help/v3.0/variable/CMAKE_CURRENT_BINARY_DIR.html
# https://cmake.org/cmake/help/v3.0/command/execute_process.html
# https://stackoverflow.com/questions/11783932/how-to-add-linker-or-compile-flag-in-cmake-file
# https://cmake.org/cmake/help/v3.2/manual/cmake.1.html#options
# https://stackoverflow.com/questions/17511496/create-a-shared-library-with-cmake
# a rant on autohell: https://stackoverflow.com/questions/4071880/what-are-the-differences-between-autotools-cmake-and-scons
# autohell: https://en.wikipedia.org/wiki/Automake
# https://www.gnu.org/software/automake/manual/html_node/Libtool-Libraries.html#Libtool-Libraries
# http://mij.oltrelinux.com/devel/autoconf-automake/
# https://eklitzke.org/how-to-autotools

