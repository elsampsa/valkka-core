from setuptools import setup, Extension, find_packages
import subprocess
import os
import copy
import sys
import numpy

custom_compilation=True   # using header and .so files that you have provided (including locally compiled libValkka.so)
# custom_compilation=False  # using system-wide installed header and .so files (including system-wide installed libValkka.so)

# *** define live555 and ffmpeg header files .. found from a custom location ***
custom_ffmpeg_include_dirs =["ext"]
custom_live555_include_dirs=["ext/liveMedia","ext/groupsock","ext/BasicUsageEnvironment","ext/UsageEnvironment"]

# *** define live555 and ffmpeg header files .. as fround from a system-wide installation ***
stock_ffmpeg_include_dirs =[]
try:
  p=subprocess.run(["pkg-config","--cflags","live555"],stdout=subprocess.PIPE)
  st=p.stdout.decode("utf-8") # from "bytes" to string
  st.replace("-I","").split()
  stock_live555_include_dirs=st.split()
except Exception as e:
  print("Could not find system installed live555 header files :",str(e))
  stock_live555_include_dirs=[]

include_dirs       =[]
include_dirs      +=["include"]
include_dirs      +=[numpy.get_include()]

if (custom_compilation):
  include_dirs    +=custom_live555_include_dirs
  include_dirs    +=custom_ffmpeg_include_dirs
else:
  include_dirs    +=stock_live555_include_dirs
  include_dirs    +=stock_ffmpeg_include_dirs

extra_compile_args =[]
# extra_compile_args+=["-O2","-DSOCKLEN_T=socklen_t","-D_LARGEFILE_SOURCE=1","-D_FILE_OFFSET_BITS=64","-Wall","-DBSD=1"]
extra_compile_args+=["-fPIC","-std=c++14"]
  
extra_link_args    =[]

if (custom_compilation):
  try:
    ld_path=os.environ["LD_LIBRARY_PATH"]
  except Exception as e:
    print("Could not access environment variable LD_LIBRARY_PATH :",str(e))
  else:
    if (ld_path.strip()!=""):
      extra_link_args +=["-L"+ld_path]
else:
  pass

libraries         =[]
libraries         +=["Valkka"]

swig_opts         =[]
swig_opts         +=["-c++","-I/usr/include"]

library_dirs      =[]

sources           =["valkka_core.i"]

ext_modules=[]
ext=Extension("_valkka_core",sources=sources,include_dirs=include_dirs,extra_compile_args=extra_compile_args,extra_link_args=extra_link_args,libraries=libraries,swig_opts=swig_opts,library_dirs=library_dirs)
ext_modules.append(ext)

# https://setuptools.readthedocs.io/en/latest/setuptools.html#basic-use
setup(
  name = "Valkka",
  # WARNING: the following line is modified by the "setver.bash" script
  version = "0.3.6", 
  install_requires = [
    'docutils>=0.3',
    'numpy>=1.14'
    ],
  packages = find_packages(),
  # scripts = ['say_hello.py'],

  include_package_data=True, # conclusion: NEVER forget this : files get included but not installed

  # "package_data" keyword is a practical joke: use MANIFEST.in instead
  
  # metadata for upload to PyPI
  author = "Sampsa Riikonen",
  author_email = "sampsa.riikonen@iki.fi",
  description = "Valkka python API",
  license = "LGPLv3+",
  # keywords = "hello world example examples",
  # url = "http://example.com/HelloWorld/",   # project home page, if any

  # could also include long_description, download_url, classifiers, etc.
  ext_modules=ext_modules
)
