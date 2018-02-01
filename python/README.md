
# Packaging

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

Run the script "make_links.bash".  Now the setup script can find the valkka header files.

Before compiling, we must inform the linker where the "libValkka.so" is.  If you are using system-wide (1), linker should find it automatically.  If you're doing custom-compiled, run first "source test_env.bash" in your relevant build directory (it sets the "LD_LIBRARY_PATH" environment variable that we use here also in link time)

Run the "compile3.bash" script (this also calls the script "make_swig_file.bash" that extracts required parts from valkka header files to build the python interface).

After this, you should have the files "valkka_core.py" and "_valkka_core.py" in the "valkka/" directory.

(between successive "compile3.bash" runs, you might want to run the "clean.bash" script)

### Create the python package

Just run "makepkg3.bash".  After that you have a python ".whl" file which you can install with "pip3 install filename.whl"

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

