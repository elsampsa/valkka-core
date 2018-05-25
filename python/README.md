
# Packaging for Python

## Building the Python3 bindings

### Choose system-wide installed vs. custom-compiled

So, you have finally created libValkka and packaged it.  Now it's time to create and package the python bindings!

You might want to create the bindings by (1) using system-wide installed header and .so files (which you installed with "sudo apt-get [whatever]-dev"), or by (2) your custom header files (this is the recommended way):

1. System-wide installed libraries and header files:
  - Set "custom_compilation=True" in "setup.py" 

2. Custom-compiled and custom header files **recommended**
  - Set "custom_compilation=False" in setup.py

### Create the extensions using SWIG

Go to directory "ext/" and run there scripts "ln_ffmpeg.bash" and "ln_live.bash".  Now the setup script can find ffmpeg and live555 header files.

Run in this directory "make_links.bash".  Now the setup script can find the valkka header files.

Before compiling, we must inform the linker where the "libValkka.so" is.  If you are using system-wide (1), linker should find it automatically.  If you're doing custom-compiled, run first "source test_env.bash" in your build directory (it sets the "LD_LIBRARY_PATH" environment variable that we use here also in link time)

Next, depending on your python version, run 

- compile34.bash for Python3.4
- compile35.bash for Python3.5

(these scripts also call the script "make_swig_file.bash" that extracts required parts from valkka header files to build the python interface).

[If you want to build packages for different python versions (other than the default python3.x of your system), you need to manage several versions of python.  See my answer to this SO question: https://stackoverflow.com/questions/2812520/pip-dealing-with-multiple-python-versions/50319252#50319252 ]

After this, you should have the files "valkka_core.py" and "_valkka_core.py" in the "valkka/" directory.

(between successive "compile3x.bash" runs, you might want to run the "clean.bash" script)

### Create the python package

Here we have several options for different distribution strategies:

1. *makesourcekg3.bash*
  This creates a traditional source package.  When you run install on a package created by this script, it tries to compile the cpp extensions.  The scenario for this one is, that you have installed libValkka.so.x from a debian package and that the header files necessary for compilation are in place - and that the setup.py finds them (say, with pkg-config).  Not functional at the moment.

2. *makecontainedpkg3.bash*
  This creates a "self-contained" package with all necessary binary files.  Python code is in source format (.py), while two binary files, the Valkka shared library (valkka/libValkka.so.N) and pre-compiled Valkka python bindings (valkka/_valkka_core.so) are included in the package.  
  Works only at a system where the package was compiled (in my case, x86_64 based on Ubuntu 16.04 LTS), or in a very similar one (same versions of all fundamental libraries).
  You need to keep a softlink to your compiled libValkka.so.N in the valkka/ directory

3. *makebindingpkg3.bash*
  This version includes only the python bindings (valkka/_valkka_core.so), not libValkka (which you have to install, say, from a debian package)

4. *makewhlpk3.bash*
  Like (3), but a traditional python binary package - very sensitive to the exact python version, etc.

Strategy (1) is the healthiest one.  However, it requires the installation of a deb package (too much effort for some people nowadays) and were just in the alpha version, so  **for the moment, use (2).**  (btw, that's what you get from PyPi).  Remember to "cleandist.bash" between successive make*pkg3.bash runs.

# Developers

## Development environment

So, you have your custom-compiled libs, C files under development etc., and meanwhile, you also want to test if the python bindings work.  You still have to run in this directory "source test_env.bash" to correctly append your PYTHONPATH

Now you have a shell where you can run your python3 test programs - and correctly find the valkka python module and libValkka.so

# Test it

```
ipython3
from valkka.valkka_core import *
```

If it can't find "libValkka.so" you did not install libValkka correctly or did not set your LD_LIBRARY_PATH

Run also the "quicktest.py" in this directory to check that everything is in place (i.e that swig generated all relevant constructors, etc.)


