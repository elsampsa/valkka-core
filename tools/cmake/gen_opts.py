#!/usr/bin/python3
"""
Generates options section for CMakeLists.txt and also default options variable for run_cmake.bash
"""

st="""
option($opt$ "$opt$" OFF)
if    ($opt$)
  add_definitions("-D$OPT$")
  message("$OPT$ ENABLED")
endif ($opt$)"""

opts=[
"VALGRIND_GPU_DEBUG", # // enable this for valgrind debugging.  Otherwise direct memory access to GPU drives it crazy.
"NO_LATE_DROP_DEBUG", # // don't drop late frame, but present everything in OpenGLThreads fifo.  Useful when debuggin with valgrind (as all frames arrive ~ 200 ms late)
#----verbosity switches----
"AVTHREAD_VERBOSE",
"DECODE_VERBOSE",
"LOAD_VERBOSE",    # // shows information on loading YUVPBO and TEX structures
"PRESENT_VERBOSE", # // enable this for verbose output about queing and presenting the frames in OpenGLThread // @verbosity       
"RENDER_VERBOSE",  # // enable this for verbose rendering
"FIFO_VERBOSE"
]

for opt in opts:
  print(st.replace("$opt$",opt.lower()).replace("$OPT$",opt.upper()))

print()

stt='options="'
for opt in opts:
  stt+="-D"+opt.lower()+"=OFF "
stt+='"'

print(stt)
