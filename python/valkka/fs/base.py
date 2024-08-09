"""
base.py : api level 1 => api level 2 encapsulation for ValkkaFS.  Base class.

 * Copyright 2017-2023 Valkka Security Ltd. and Sampsa Riikonen
 * Copyright 2024 Sampsa Riikonen
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

@brief   api level 1 => api level 2 encapsulation for ValkkaFS.  Base class.
"""
import time
import traceback
import json
import os
import glob
import numpy
import logging
from valkka import core
# from valkka.api2.tools import *
from valkka.api2.tools import parameterInitCheck, getLogger, setLogger
from valkka.fs.tools import findBlockDevices, formatMstimestamp, formatMstimeTuple
from valkka.api2.exceptions import ValkkaFSLoadError
import datetime
import traceback


class ValkkaFS:
    """ValkkaFS instance keeps the book for the block file system.
    
    A single instance refers to one block file system and can be shared between one writer and several reader instances, so it is thread safe.
    
    - Instantiate ValkkaFS with the static methods (loadFromDirectory or newFromDirectory) and then pass them to (api level 1) ValkkaFSWriter and ValkkaFSReader instances
    - Can be used with FGroups and Managers (see group.py and manager.py)
    
    json file with:
    
    - All the parameters of this class
    - The state of the recording (latest block)
    - Let's use a dedicated directory for this
    
    ::
    
        directory_name/
            valkkafs.json
            dumpfile
            blockfile
        
    """
    core_valkkafs_class = core.ValkkaFS

    @classmethod
    def loadFromDirectory(cls, dirname, verbose=False):
        jsonfile = os.path.join(dirname,"valkkafs.json")
        #assert(os.path.exists(dirname))
        #assert(os.path.exists(jsonfile))
        if not os.path.exists(dirname):
            raise ValkkaFSLoadError("no such dir "+dirname)
        if not os.path.exists(jsonfile):
            raise ValkkaFSLoadError("no such file "+jsonfile)

        f=open(jsonfile, "r")
        dic=json.loads(f.read())
        f.close()

        partition_uuid  = dic["partition_uuid"]
        dumpfile        = dic["dumpfile"]
        blockfile       = dic["blockfile"]
        blocksize       = dic["blocksize"]
        n_blocks        = dic["n_blocks"]
        current_block   = dic["current_block"]
        # either partition_uuid or local file
        # assert((partition_uuid is None) or (dumpfile is None)) # nopes .. dumpfile refers to the correct /dev/name taken earlier from partition_uuid
        
        if partition_uuid is None:
            if not os.path.exists(dumpfile):
                raise ValkkaFSLoadError("no such dumpfile "+dumpfile)
            #assert(os.path.exists(dumpfile))
        
        if not os.path.exists(blockfile):
            raise ValkkaFSLoadError("missing blockfile "+blockfile)
        if current_block > n_blocks:
            raise ValkkaFSLoadError("Inconstent FS: current_block, n_blocks "
                +str(current_block)+", "+str(n_blocks))
        #assert(os.path.exists(blockfile))
        #assert(current_block < n_blocks)
        # fs = ValkkaFS(
        fs = cls(
            partition_uuid = partition_uuid,
            dumpfile       = dumpfile, 
            blockfile      = blockfile,
            blocksize      = blocksize,
            n_blocks       = n_blocks,
            current_block  = current_block,
            jsonfile       = jsonfile,
            verbose        = verbose
            )
        
        fs.reload_()
        return fs
        
        
    @classmethod
    def newFromDirectory(cls, dirname=None, blocksize=None, n_blocks=None, device_size=None, partition_uuid=None, verbose=False):
        """If partition_uuid is defined, then use a dedicated block device
        """
        assert(isinstance(dirname, str))
        
        jsonfile  =os.path.join(dirname,"valkkafs.json")
        blockfile =os.path.join(dirname,"blockfile")
        dumpfile  =os.path.join(dirname,"dumpfile")

        if (os.path.exists(dirname)): # clear directory
            if os.path.exists(jsonfile)  : os.remove(jsonfile)
            if os.path.exists(blockfile) : os.remove(blockfile)
            if os.path.exists(dumpfile)  : os.remove(dumpfile)
        else:
            os.makedirs(dirname)
        
        assert(isinstance(blocksize, int))
        
        blocksize = max(core.FS_GRAIN_SIZE, (blocksize - blocksize%core.FS_GRAIN_SIZE)) # blocksize must be a multiple of 512
        
        if (isinstance(n_blocks, int)):
            pass
        else: # if number of blocks is not defined, we need the devicefile size
            assert(isinstance(device_size, int))
            n_blocks = device_size // blocksize

        if (partition_uuid is not None): # if no partition has been defined, then use default filename "dumpfile" in the directory
            dumpfile = None
        
        print("ValkkaFS: newFromDirectory: %s, %s" % (str(dumpfile), str(partition_uuid)))
        
        fs = cls(
            partition_uuid = partition_uuid,
            dumpfile   = dumpfile,
            blockfile  = blockfile,
            blocksize  = blocksize,
            n_blocks   = n_blocks,
            current_block = 0,
            jsonfile   = jsonfile,
            verbose    = verbose
            )
        
        fs.reinit() # stripe the device
        return fs
        
        
    @staticmethod
    def checkDirectory(dirname):
        jsonfile  =os.path.join(dirname,"valkkafs.json")
        assert(os.path.exists(dirname)), "no such directory "+dirname
        assert(os.path.exists(jsonfile)), "no such jsonfile "+jsonfile
        f=open(jsonfile, "r")
        dic=json.loads(f.read())
        f.close()
        
        partition_uuid  = dic["partition_uuid"]
        dumpfile        = dic["dumpfile"]
        blockfile       = dic["blockfile"]
        blocksize       = dic["blocksize"]
        n_blocks        = dic["n_blocks"]
        current_block   = dic["current_block"]
        
        block_device_dic = findBlockDevices()
        
        if (partition_uuid is not None):
            assert(partition_uuid.lower() in block_device_dic) # check that partition_uuid exists
            # print(block_device_dic, dumpfile)
            assert(block_device_dic[partition_uuid.lower()][0] == dumpfile) # check that it correspond to the correct device
        
             
    filesystem_defs={
        "dumpfile"  : None,             # frames are dumped here
        "partition_uuid" : None,        # .. or into a partition with this uuid
        "blockfile" : str,              # block book-keeping in this file
        "blocksize" : int,              # size of a single block
        "n_blocks"  : int,              # number of blocks
        "current_block" : (int, 0),     # writing is resumed at this block
        "jsonfile"  : str               # this gets rewritten at each block
        }
    
    parameter_defs={
        "verbose"  : (bool, False)
        }
    
    parameter_defs.update(filesystem_defs)
    
    instance_counter = 0 # a class variable

    def __init__(self, **kwargs):
        """If partition_uuid defined, the dumpfile is set by ctor to the device defined by partition_uuid
        """
        self.pre = self.__class__.__name__ + " : "
        # checks kwargs agains parameter_defs, attach ok'd parameters to this
        # object as attributes
        logname = self.__class__.__name__
        parameterInitCheck(ValkkaFS.parameter_defs, kwargs, self)

        self.name = str(self.__class__.instance_counter)
        self.__class__.instance_counter += 1
        logname = logname + " " + self.name
        self.logger = getLogger(logname)
        # setLogger(self.logger, logging.DEBUG) # TODO: unify the verbosity/logging somehow

        self.block_cb = None # a custom callback in the callback chain when a new block is ready
        
        block_device_dic = findBlockDevices() 
        
        if (self.dumpfile==None): # no dumpfile defined .. so it must be the partition uuid
            assert(isinstance(self.partition_uuid, str))

            if self.partition_uuid in block_device_dic:
                pass
            else:
                self.logger.debug("block devices %s", block_device_dic)
                raise(AssertionError("could not find block device"))
            self.dumpfile = block_device_dic[self.partition_uuid.lower()][0]
        else:
            assert(isinstance(self.dumpfile, str))
        
        self.logger.debug("ValkkaFS: dumpfile = %s", str(self.dumpfile))
        
        # self.core = core.ValkkaFS(self.dumpfile, self.blockfile, self.blocksize, self.n_blocks, False) # dumpfile, blockfile, blocksize, number of blocks, init blocktable
        self.core = self.core_valkkafs_class(self.dumpfile, self.blockfile, self.blocksize, self.n_blocks, False) # dumpfile, blockfile, blocksize, number of blocks, init blocktable
        
        self.blocksize = self.core.getBlockSize() # in the case it was adjusted ..
        
        # self.blocktable_ = numpy.zeros((self.core.get_n_blocks(), self.core.get_n_cols()),dtype=numpy.int_) # this memory area is accessed by cpp
        self.blocktable  = numpy.zeros((self.core.get_n_blocks(), self.core.get_n_cols()),dtype=numpy.int_) # copy of the cpp blocktable
        # self.getBlockTable()
        self.core.setBlockCallback(self.new_block_cb__) # callback when a new block is registered
        """
        see valkkafs.cpp: 
            ValkkaFS::setBlockCallback
            ValkkaFS::writeBlock
        """
        self.logger.debug("ValkkaFS: resuming writing at block %s", self.current_block)
        self.core.setCurrentBlock(self.current_block)
        # # attach an analysis tool
        # self.analyzer = core.ValkkaFSTool(self.core)
        # self.verbose = True

    def setLogLevel(self, level):
        getLogger(self.logger, level = level)

    def __str__(self):
        if self.dumpfile is None:
            return "<ValkkaFS %s>" % (self.partition_uuid)
        else:
            return "<ValkkaFS %s>" % (self.dumpfile)

    def getName(self):
        return self.name

    def update(self):
        self.new_block_cb__(True, "init")


    def is_same(self, **dic):
        """Compare this ValkkaFS to given parameters

        Use parameters that are not modified by the ctor
        """
        self.logger.debug("partition_uuid: %s %s",self.partition_uuid,dic["partition_uuid"])
        self.logger.debug("blocksize     : %s %s",self.blocksize, dic["blocksize"])
        self.logger.debug("n_blocks      : %s %s", self.n_blocks, dic["n_blocks"])

        return (\
        (self.partition_uuid  == dic["partition_uuid"]) and \
        (self.blocksize       == dic["blocksize"]) and \
        (self.n_blocks        == dic["n_blocks"]) \
        )


    def new_block_cb__(self, propagate, par):
        """input tuple elements:
        
        boolean    : should the callback be propagated or not
        int / str  : int = block number, str = error message
        
        Called as a callback from the cpp side, see:

        ::

            valkkafs.cpp: 
                ValkkaFS::writeBlock

        """
        #propagate = tup[0]
        #par = tup[1]
        try:
            self.logger.debug("new_block_cb__: %s, propagate: %s par: %s", str(self), propagate, par)
            if isinstance(par, int):
                self.current_block = par
                self.logger.debug("new_block_cb__: got block num %s", par)
                self.writeJson()
                self.logger.debug("new_block_cb__: wrote json")
            elif isinstance(par, str):
                self.logger.debug("new_block_cb__: got message: %s", par)

            self.getBlockTable() # update BT values here at the python side

            if (self.block_cb is not None) and propagate:
                self.logger.debug("new_block_cb: subcallback")
                self.block_cb()
                
            self.logger.debug("new_block_cb: bye")
                
        except Exception as e:
            self.logger.debug("failed with '%s'", e)
            traceback.print_exc()
        
                
    def setBlockCallback(self, cb):
        self.block_cb = cb
        
            
    def report(self):
        print(self.pre)
        for par in ValkkaFS.filesystem_defs:
            print(self.pre, par, getattr(self, par))
        print(self.pre)
            

    def writeJson(self):
        """Internal
        """
        dic={}
        jsonf = open(self.jsonfile,"w")
        for par in ValkkaFS.filesystem_defs:
            dic[par] = getattr(self, par)
        jsonf.write(json.dumps(dic)+"\n")


    def getPars(self):
        # TODO: lock protection..? if we're in the middle of changing self.current_block .. on the other hand, python gil should take care of that
        dic={}
        jsonf = open(self.jsonfile,"w")
        for par in ValkkaFS.parameter_defs:
            dic[par] = getattr(self, par)
        return dic
        
        
    def reinit(self):
        # TODO: remove dumpfile
        # verbose=True
        verbose=False
        self.core.clearTable()
        self.core.dumpTable()
        self.writeJson()
        # for regular, files, write them through
        # for block devices, just write the stripes
        if (self.dumpfile.find("/dev/")>-1): # this is a dedicated device
            self.logger.debug("ValkkaFS: striping device")
            self.core.clearDevice(False, verbose) # just write long int zeros in the beginning of each block
            self.logger.debug("ValkkaFS: striped device")
        else:
            self.logger.debug("ValkkaFS: clearing device")
            self.core.clearDevice(True, verbose) # write-through the file
            self.logger.debug("ValkkaFS: cleared device")
        
        
    def reload_(self):
        self.core.readTable()
    
    
    def getBlockTable(self, reread=False):
        """Updates self.blocktable & returns it
        """
        self.logger.debug("ValkkaFS: getBlockTable")
        # traceback.print_stack()
        if reread:
            # normally not required since this is call originates
            # from a callback indicating a new block
            self.core.updateTable() 
        # self.core.setArrayCall(self.blocktable_) # copy data from cpp to python (this is thread safe)
        self.core.setArrayCall(self.blocktable) # copy data from cpp to python (this is thread safe)
        # self.blocktable[:,:] = self.blocktable_[:,:] # copy the data .. self.blocktable can be accessed without fear of being overwritten during accesss
        # return self.blocktable
        self.logger.debug("ValkkaFS: getBlockTable: exit")
        # return self.blocktable_
        return self.blocktable
    

    def getTimeRange(self):
        """Return full timerange of frames in the blocktable

        Remember blocktable wrapping
        """
        raise(BaseException("virtual"))

    
    def getInd(self, times):
        """Times is a tuple of mstimestamps.  Returns blocks between mstimestamps
        
        Remember blocktable wrapping
        """
        raise(BaseException("virtual"))


    def getIndNeigh(self, n=1, time=0):
        """Get indices of neighboring blocks if block n

        Remember blocktable wrapping
        """
        raise(BaseException("virtual"))





