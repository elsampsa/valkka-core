"""
__init__.py : Valkka python bindings module constructor

 * Copyright 2017, 2018 Valkka Security Ltd. and Sampsa Riikonen.
 * 
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 * 
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>
 *
 */

@file    __init__.py
@author  Sampsa Riikonen
@date    2017
@version 0.15.0 
  
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

from .valkka_core import * # import everything to valkka.core namespace
__version__=str(VERSION_MAJOR)+"."+str(VERSION_MINOR)+"."+str(VERSION_PATCH)


