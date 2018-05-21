from setuptools import setup, Extension, find_packages
from setuptools.command.install import install
import subprocess
import os
import copy
import sys
import numpy

"""
class CustomInstallCommand(install):
  # Customized setuptools install command
  def run(self):
    print("Custom install command")
    # self.skip_build=True
    #print(dir(self))
    install.run(self)
"""

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

runtime_library_dirs =["$ORIGIN"] # https://stackoverflow.com/questions/9795793/shared-library-dependencies-with-distutils

ext_modules=[]

ext=Extension("_valkka_core",sources=sources,include_dirs=include_dirs,extra_compile_args=extra_compile_args,extra_link_args=extra_link_args,libraries=libraries,swig_opts=swig_opts,library_dirs=library_dirs,runtime_library_dirs=runtime_library_dirs)
ext_modules.append(ext)

"""
Now follows a HACK that requires some explanation..

1) At the build stage, we want setuptools to include ext_modules and compile them

2) At the packaging stage (python3 setup.py sdist) we want setuptools to include a setup.py into the package that has no knowledge of ext_modules ..
.. this way we avoid building the binary libraries at the installation stage (done typically with pip), at the end-users linux box

The line after this comment section is modded using sed at the different stages of building the package

Why all this?

* There's no way we can build whole libValkka from scratch at the end-users linux box
* .whl distributions are over-sensitive to the python 3.x version (3.4, 3.5, 3.6, etc.) because they package python bytecode
* .. they shouldn't be sensitive to that, but instead sensitive simply to the architecture (i686 vs. x86_64, etc.)
* We need a binary package where the python part can be python source, but there are binary, pre-compiled .so files.   There seems to be no way of handling this in the current python packaging system.

An alternative solution would be ..

* End-user install separately .deb that has libValkka.so.0 and the necessary header files
* .. after that, this can be built at the end-users linux box
"""
# ext_modules=[] # SWITCH

here = os.path.abspath(os.path.dirname(__file__))
try:
  f=open(os.path.join(here, 'PYPI.rst'),encoding='utf-8')
except:
  print("could not open PYPI.rst")
  long_description=""
else:
  long_description = f.read()
  f.close()
  
# https://setuptools.readthedocs.io/en/latest/setuptools.html#basic-use
setup(
  #cmdclass={
  #  'install': CustomInstallCommand,
  #  },
  
  name = "valkka",
  
  # WARNING: the following line is modified by the "setver.bash" script
  version = "0.4.4", 
  
  install_requires = [
    'docutils>=0.3',
    'numpy>=1.14'
    'PyQt5>=5.10',
    'imutils>=0.4'
    ],
  packages = find_packages(),
  # scripts = ['say_hello.py'],

  include_package_data=True, # conclusion: NEVER forget this : files get included but not installed
  # note: the "package_data" keyword is a practical joke: use MANIFEST.in instead
  
  # metadata for upload to PyPI
  author           = "Sampsa Riikonen",
  author_email     = "sampsa.riikonen@iki.fi",
  description      = "OpenSource Video Surveillance and Management for Linux",
  long_description =long_description,
  # long_description_content_type='text/markdown', # this does not work
  license          = "AGPLv3+",
  url              = "https://github.com/elsampsa/valkka-core",   # project home page, if any
  keywords         = "video surveillance vision camera streaming live",
  classifiers      =[  # Optional
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',
    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Operating System :: POSIX :: Linux',
    'Topic :: Multimedia :: Video',
    # Pick your license as you wish
    'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
  project_urls={
    'Tutorial': 'https://elsampsa.github.io/valkka-examples/'
  },
  ext_modules=ext_modules
)
