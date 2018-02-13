# Cmake file for Valkka library
# (C) 2017 Sampsa Riikonen
#
# cmake .
# make
# (or make VERBOSE=1)
#

cmake_minimum_required(VERSION 3.1)
find_package(PkgConfig REQUIRED)

project(Valkka)

# **** INPUT VARIABLES ****
option(custom_build "custom_build" ON )
option(use_shared   "use_shared"   OFF)
option(ffmpeg_root  "ffmpeg_root"  "")
option(live555_root "live555_root" "")

SET(FFMPEG_ROOT ${ffmpeg_root})
SET(LIVE555_ROOT ${live555_root})

if    (custom_build)
  SET(CUSTOM_BUILD "TRUE")
else  (custom_build)
  SET(CUSTOM_BUILD "FALSE")
endif (custom_build)

if    (use_shared)
  SET(USE_SHARED "TRUE")
else  (use_shared)
  SET(USE_SHARED "FALSE")
endif (use_shared)


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



# ****************************************************************
# WARNING: the following three lines are modified by the "setver.bash" script
SET(MAJOR_VERSION "0") 
SET(MINOR_VERSION "3") 
SET(PATCH_VERSION "0") 
set(VERSION_STRING ${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION})

# *** define build type: Debug or Release # now from the command line
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

# *** define if live555 and ffmpeg are .. # now from the command line
# set(CUSTOM_BUILD "TRUE") # .. custom-compiled **RECOMMENDED**  You might have to edit the library versions: see tag [LIBRARY_VERSIONS]
# set(CUSTOM_BUILD "FALSE") # .. system-wide installed header and .so files (i.e. with "apt-get install ..")

# *** if you are using custom compiled, choose either .. # now from the command line
# set(USE_SHARED "TRUE") # .. dependencies of Valkka (live555 and ffmpeg) are found from other dynamic libraries 
# set(USE_SHARED "FALSE") # .. live555 and ffmpeg are baked into Valkka as static libraries **RECOMMENDED**

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
# pkg_search_module(PYTHON REQUIRED python3)
# TODO: python3

# compiler flags
add_compile_options("-std=c++14" "-pthread") #  ${GL_CFLAGS})

# [define library source files]
file(GLOB SOURCES src/*.cpp)

# [we're compiling a library here..]
add_library(${PROJECT_NAME} SHARED ${SOURCES})
# => not the target_* commands work

# [define library header files]
# include_directories(include)
target_include_directories(${PROJECT_NAME} PUBLIC include)

# [set shared library version]
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${VERSION_STRING} SOVERSION ${MAJOR_VERSION})

# [add openGL dependencies]
target_include_directories(${PROJECT_NAME} PUBLIC ${X11_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS}) # ${PYTHON_INCLUDE_DIRS})
# # target_link_libraries(${PROJECT_NAME} ${GL_LIBS})
target_link_libraries(${PROJECT_NAME} ${X11_LIBRARIES} ${GLEW_LIBRARIES} ${ALSA_LIBRARIES})

if(CUSTOM_BUILD) # [CUSTOM]
  message("USING CUSTOM COMPILED LIVE555 AND FFMPEG - GOOD")
  message("LIVE555 ROOT: ${LIVE555_ROOT}")
  message("FFMPEG  ROOT: ${FFMPEG_ROOT}")
  # include_directories("include/ext/liveMedia" "include/ext/groupsock" "include/ext/BasicUsageEnvironment" "include/ext/UsageEnvironment")
  # include_directories("include/ext")
  # target_include_directories(${PROJECT_NAME} PUBLIC "include/ext/liveMedia" "include/ext/groupsock" "include/ext/BasicUsageEnvironment" "include/ext/UsageEnvironment")
  # target_include_directories(${PROJECT_NAME} PUBLIC "include/ext")
  # # now defined from command line:
  target_include_directories(${PROJECT_NAME} PUBLIC "${LIVE555_ROOT}/liveMedia/include" "${LIVE555_ROOT}/groupsock/include" "${LIVE555_ROOT}/BasicUsageEnvironment/include" "${LIVE555_ROOT}/UsageEnvironment/include")
  target_include_directories(${PROJECT_NAME} PUBLIC "${FFMPEG_ROOT}")
  
  # target_link_libraries(${PROJECT_NAME} "-Llib")
  # link_directories("-Llib")
  target_link_libraries(${PROJECT_NAME} "-L${CMAKE_CURRENT_SOURCE_DIR}/lib")
  
  # .. now the include directories have the correct absolute paths
  if(USE_SHARED) # custom-compiled shared libraries [LIBRARY_VERSIONS]
    message("USING EXTERNAL SHARED DEPENDENCIES FOR LIVE555 AND FFMPEG - ARE YOU SURE?")
    target_link_libraries(${PROJECT_NAME} ":libliveMedia.so.52" ":libgroupsock.so.8" ":libBasicUsageEnvironment.so.1" ":libUsageEnvironment.so.3") # live555
    target_link_libraries(${PROJECT_NAME} ":libavcodec.so.57" ":libavformat.so.57" ":libavutil.so.55" ":libswscale.so.4") # ffmpeg
    # set(LIVE_LIBS   "-l:libliveMedia.so.52 -l:libgroupsock.so.8 -l:libBasicUsageEnvironment.so.1 -l:libUsageEnvironment.so.3") # live555
    # set(FFMPEG_LIBS "-l:libavcodec.so.57  -l:libavformat.so.57  -l:libavutil.so.55 -l:libswscale.so.4") # ffmpeg
  else(USE_SHARED) # custom-compiled static libraries
    message("USING STATICALLY COMPILED LIVE555 AND FFMPEG - GOOD")
    # target_link_libraries(${PROJECT_NAME} "vda") # could this fix the missing "vaUnmapBuffer", etc. crap?
    target_link_libraries(${PROJECT_NAME} "rt")
    target_link_libraries(${PROJECT_NAME} "-Wl,--allow-multiple-definition" "-Wl,-Bsymbolic" "-Wl,--start-group" "-Wl,--whole-archive")
    target_link_libraries(${PROJECT_NAME} ":libliveMedia.a" ":libgroupsock.a" ":libBasicUsageEnvironment.a" ":libUsageEnvironment.a")
    # target_link_libraries(${PROJECT_NAME} ":libavdevice.a" ":libavfilter.a" ":libavformat.a" ":libavcodec.a" ":libavutil.a" ":libswscale.a" ":libswresample.a")
    target_link_libraries(${PROJECT_NAME} ":libavfilter.a" ":libavformat.a" ":libavcodec.a" ":libavutil.a" ":libswscale.a" ":libswresample.a")
    target_link_libraries(${PROJECT_NAME} "-Wl,--no-whole-archive" "-Wl,--end-group")
    
    # add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND "objcopy" ARGS "-N" "vaUnmapBuffer" $<TARGET_FILE:${PROJECT_NAME}>)
                   
  endif(USE_SHARED)

else(CUSTOM_BUILD) # system-provided shared libraries
  message("USING SYSTEM PROVIDED SHARED LIBRARIES FOR LIVE555 AND FFMPEG - ARE YOU SURE?")
  # live555:
  #exec_program(pkg-config ARGS --cflags live555 OUTPUT_VARIABLE LIVE_CFLAGS)
  #exec_program(pkg-config ARGS --libs live555 OUTPUT_VARIABLE LIVE_LIBS)
  # ffmpeg:
  #exec_program(pkg-config ARGS --cflags libavcodec libavformat libavutil libswscale OUTPUT_VARIABLE FFMPEG_CFLAGS)
  #exec_program(pkg-config ARGS --libs libavcodec libavformat libavutil libswscale OUTPUT_VARIABLE FFMPEG_LIBS)
  
  # # ** how to add the -I switches for from exec_program => pkg-config ..?
  # # add_compile_options(${LIVE_CFLAGS} ${FFMPEG_CFLAGS}) # -I switches through compile options here.. # does not work ..?
  # set(CMAKE_CXX_FLAGS "-std=c++14 -pthread ${LIVE_CFLAGS} ${FFMPEG_CFLAGS} ${GL_CFLAGS}") # .. so let's do it like this
  # # like this:
  # # target_include_directories(${PROJECT_NAME} PUBLIC ${LIVE_CFLAGS} ${FFMPEG_CFLAGS}) # nopes ..
  # target_link_libraries(${PROJECT_NAME} ${FFMPEG_LIBS} ${LIVE_LIBS})
  
  pkg_search_module(LIVE555 REQUIRED live555)
  pkg_search_module(LIBAVCODEC REQUIRED libavcodec)
  pkg_search_module(LIBAVFORMAT REQUIRED libavformat)
  pkg_search_module(LIBAVUTIL REQUIRED libavutil)
  pkg_search_module(LIBSWSCALE REQUIRED libswscale)
  
  target_include_directories(${PROJECT_NAME} PUBLIC ${LIVE555_INCLUDE_DIRS})
  target_include_directories(${PROJECT_NAME} PUBLIC ${LIBAVCODEC_INCLUDE_DIRS})
  target_include_directories(${PROJECT_NAME} PUBLIC ${LIBAVFORMAT_INCLUDE_DIRS})
  target_include_directories(${PROJECT_NAME} PUBLIC ${LIBAVUTIL_INCLUDE_DIRS})
  target_include_directories(${PROJECT_NAME} PUBLIC ${LIBSWSCALE_INCLUDE_DIRS})
  
  target_link_libraries(${PROJECT_NAME} ${LIVE555_LIBRARIES})
  target_link_libraries(${PROJECT_NAME} ${LIBAVCODEC_LIBRARIES})
  target_link_libraries(${PROJECT_NAME} ${LIBAVFORMAT_LIBRARIES})
  target_link_libraries(${PROJECT_NAME} ${LIBAVUTIL_LIBRARIES})
  target_link_libraries(${PROJECT_NAME} ${LIBSWSCALE_LIBRARIES})
  
endif(CUSTOM_BUILD)

# *** test programs ***
# use the script "test/list.bash" to update this list

set(TESTNAMES "av_live_thread_test" "av_thread_test" "classes_test" "fifo_test" "file_test" "frames_test" "live_stream_check_test" "live_streaming_test" "live_thread_test" "log_test" "opengl_test" "opengl_thread_test" "render_test" "shmem_test" "swscale_test" "threads_test")

# set(TESTNAMES "shmem_test" "swscale_test")
# set(TESTNAMES "file_test")


add_custom_target(tests) # Note: without 'ALL'
foreach( testname ${TESTNAMES} )
  add_executable(${testname} "test/${testname}.cpp") # Note: without 'ALL'
  target_include_directories(${testname} PUBLIC "include")
  
  if(CUSTOM_BUILD)
    # target_include_directories(${testname} PUBLIC "include/ext/liveMedia" "include/ext/groupsock" "include/ext/BasicUsageEnvironment" "include/ext/UsageEnvironment")
    # target_include_directories(${testname} PUBLIC "include/ext")
    target_include_directories(${testname} PUBLIC "${LIVE555_ROOT}/liveMedia/include" "${LIVE555_ROOT}/groupsock/include" "${LIVE555_ROOT}/BasicUsageEnvironment/include" "${LIVE555_ROOT}/UsageEnvironment/include")
    target_include_directories(${testname} PUBLIC "${FFMPEG_ROOT}")
  else(CUSTOM_BUILD)
    target_include_directories(${testname} PUBLIC ${LIVE555_INCLUDE_DIRS})
    target_include_directories(${testname} PUBLIC ${LIBAVCODEC_INCLUDE_DIRS})
    target_include_directories(${testname} PUBLIC ${LIBAVFORMAT_INCLUDE_DIRS})
    target_include_directories(${testname} PUBLIC ${LIBAVUTIL_INCLUDE_DIRS})
    target_include_directories(${testname} PUBLIC ${LIBSWSCALE_INCLUDE_DIRS})
  endif(CUSTOM_BUILD)  
  
  # target_include_directories(${testname} PUBLIC ${PYTHON_INCLUDE_DIRS})
  # target_link_libraries(${testname} ${PYTHON_LIBRARIES})
  
  target_link_libraries(${testname} "Valkka.so")
  target_link_libraries(${testname} "-L${CMAKE_CURRENT_BINARY_DIR}/lib")
  # target_link_libraries(${testname} "-Wl,--unresolved-symbols=ignore-all") # even when all external libraries from ffmpeg have been dropped some stupid symbols persist, "vaUnmapBuffer", etc.
  # .. but then it will bust when using the python bindings
  # .. all that crap comes from libavdevice
  
  add_dependencies(tests ${testname}) # tests depende on the executable
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

# SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.3.1-6), libgcc1 (>= 1:3.4.2-12)") 
SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libgcc1(>= 1:6.0.1), libc6(>= 2.23), libgl1-mesa-glx(>= 12.0.6), libx11-6(>= 2:1.6.3), libstdc++6(>= 5.4.0), libc6(>= 2.23), libglew1.13(>= 1.13.0)") # ubuntu 16

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
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include" DESTINATION valkka FILES_MATCHING PATTERN "*.h") # install header files
# The install command: https://cmake.org/cmake/help/v3.0/command/install.html

if(CUSTOM_BUILD) # include header files ..
  # install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include/valkka FILES_MATCHING PATTERN "*.h")
  # install(DIRECTORY "${LIVE555_ROOT}/liveMedia/include" "${LIVE555_ROOT}/groupsock/include" "${LIVE555_ROOT}/BasicUsageEnvironment/include" "${LIVE555_ROOT}/UsageEnvironment/include" DESTINATION include/valkka FILES_MATCHING PATTERN "*.hh")
  install(DIRECTORY "${LIVE555_ROOT}/liveMedia/include" DESTINATION valkka/liveMedia FILES_MATCHING PATTERN "*.h*")
  install(DIRECTORY "${LIVE555_ROOT}/groupsock/include" DESTINATION valkka/groupsock FILES_MATCHING PATTERN "*.h*")
  install(DIRECTORY "${LIVE555_ROOT}/BasicUsageEnvironment/include" DESTINATION valkka/BasicUsageEnvironment FILES_MATCHING PATTERN "*.h*")
  install(DIRECTORY "${LIVE555_ROOT}/UsageEnvironment/include" DESTINATION valkka/UsageEnvironment FILES_MATCHING PATTERN "*.h*")
  
  # include header files if someone want's to use the cpp API.  Must take care thought, that the API user uses the correct header files (i.e. these, instead of some other versions)
  install(DIRECTORY "${FFMPEG_ROOT}/libavfilter" DESTINATION valkka/ffmpeg FILES_MATCHING PATTERN "*.h") # don't really need this ..
  install(DIRECTORY "${FFMPEG_ROOT}/libavformat" DESTINATION valkka/ffmpeg FILES_MATCHING PATTERN "*.h")
  install(DIRECTORY "${FFMPEG_ROOT}/libavcodec" DESTINATION valkka/ffmpeg FILES_MATCHING PATTERN "*.h")
  install(DIRECTORY "${FFMPEG_ROOT}/libavutil" DESTINATION valkka/ffmpeg FILES_MATCHING PATTERN "*.h")
  install(DIRECTORY "${FFMPEG_ROOT}/libswscale" DESTINATION valkka/ffmpeg FILES_MATCHING PATTERN "*.h")
  install(DIRECTORY "${FFMPEG_ROOT}/libswresample" DESTINATION valkka/ffmpeg FILES_MATCHING PATTERN "*.h")
  
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
  
endif(CUSTOM_BUILD)

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

