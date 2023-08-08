"""
__init__.py : Valkka python bindings module constructor

 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>
 *
 */

@file    __init__.py
@author  Sampsa Riikonen
@date    2017
@version 1.5.2 
  
@brief Valkka python bindings module constructor

@section DESCRIPTION

Welcome to valkka Python bindings!

After using the "compile3.bash" script, you should have the files

_valkka_core.so
valkka_core.py

in this directory.

These are the "low-level" APIs that were created with swig directly from cpp.  You can use them with, say:

from valkka.core.valkka_core import *

.. and then create here, under the valkka module, some higher level APIs if necessary
"""

# from valkka.core.valkka_core import VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH
# __version__=str(VERSION_MAJOR)+"."+str(VERSION_MINOR)+"."+str(VERSION_PATCH)


"""Setup environmental variables used by various libraries
"""

import os

# VAAPI hw acceleration to the opensource version:
va_driver = os.environ.get("VALKKA_LIBVA_DRIVER_NAME", "i965")
if va_driver != "i965":
    print("\nWARNING: libValkka: using VAAPI driver", va_driver, "however i965 recommended\n")
os.environ["LIBVA_DRIVER_NAME"] = va_driver
# os.system("vainfo") # seems to work
#
# OpenGL: disable vsync for intel:
os.environ["vblank_mode"]="0"
# OpenGL: disable vsync for nvidia proprietary drivers:
os.environ["__GL_SYNC_TO_VBLANK"]="0"

# check that we are in a desktop environment - if not, print a warning!
if "XDG_SESSION_TYPE" in os.environ:
    if os.environ["XDG_SESSION_TYPE"] == "x11":
        pass # all good!
    else:
        print("\nWARNING: libValkka: env variable XDG_SESSION_TYPE is '"+os.environ["XDG_SESSION_TYPE"]+"' instead of the preferred 'x11'")
        print("This means that you are most likely not in a desktop session and anything related to Qt might not work\n")
else:
    print("\nWARNING: libValkka: the env variable XDG_SESSION_TYPE is missing")
    print("Qt desktop infrastructure might not work\n")

from .valkka_core import * # import everything to valkka.core namespace
__version__=str(VERSION_MAJOR)+"."+str(VERSION_MINOR)+"."+str(VERSION_PATCH)
