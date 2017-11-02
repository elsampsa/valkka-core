"""
__init__.py : Valkka python bindings module constructor

Copyright 2017 Sampsa Riikonen and Petri Eranko.

Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>

This file is part of Valkka library.

Valkka is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Valkka is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Valkka.  If not, see <http://www.gnu.org/licenses/>. 

@file    NAME.py
@author  Sampsa Riikonen
@date    2017
@version 0.1
  
@brief Valkka python bindings module constructor

@section DESCRIPTION

Welcome to valkka Python bindings!

After using the "compile3.bash" script, you should have the files

_valkka_core.so
valkka_core.py

in this directory.

These are the "low-level" APIs that were created with swig directly from cpp.  You can use them with, say:

from valkka.valkka_core import *

.. and then create here, under the valkka module, some higher level APIs if necessary
"""
