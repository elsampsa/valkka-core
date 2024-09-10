"""
thread.py : api level 1 => api level 2 encapsulation for ValkkaFSThreads

 * (c) Copyright 2017-2024 Sampsa Riikonen
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

@file    base.py
@author  Sampsa Riikonen
@date    2021
@version 1.6.1 

@brief   api level 1 => api level 2 encapsulation for ValkkaFSThreads
"""
import time
import json
import os
import glob
import numpy
import logging
from valkka import core
# from valkka.api2.tools import *
from valkka.api2.tools import parameterInitCheck
from valkka.api2.exceptions import ValkkaFSLoadError
from valkka.fs.base import ValkkaFS
import datetime
import traceback

    
class ValkkaFSWriterThread:

    parameter_defs={
        "name"      : (str, "writer_thread"),
        "valkkafs"  : ValkkaFS,
        "affinity"  : (int, -1),
        "verbose"   : (bool, False)
        }

    def __init__(self, **kwargs):
        # auxiliary string for debugging output
        self.pre = self.__class__.__name__ + " : "
        # checks kwargs agains parameter_defs, attach ok'd parameters to this
        # object as attributes
        parameterInitCheck(ValkkaFSWriterThread.parameter_defs, kwargs, self)
        self.core = core.ValkkaFSWriterThread(self.name, self.valkkafs)
        self.core.setAffinity(self.affinity)
        self.input_filter = self.core.getFrameFilter()
        self.active = True
        self.core.startCall()

    def getInput(self):
        return self.input_filter

    def close(self):
        if not self.active:
            return
        if (self.verbose):
            print(self.pre, "stopping core.ValkkaFSWriterThread")
        self.core.stopCall()
        self.active = False

    def requestClose(self):
        self.core.requestStopCall()
        
    def waitClose(self):
        self.core.waitStopCall()
        self.active = False

    def __del__(self):
        self.close()

