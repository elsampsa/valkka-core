"""
valkkafs.py : api level 1 => api level 2 encapsulation for ValkkaFS and ValkkaFSThreads

 * Copyright 2018 Valkka Security Ltd. and Sampsa Riikonen.
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

@file    valkkafs.py
@author  Sampsa Riikonen
@date    2017
@version 0.9.0 

@brief   api level 1 => api level 2 encapsulation for ValkkaFS and ValkkaFSThreads
"""
import time
import json
import os
import glob
import numpy
from valkka import core
from valkka.api2.tools import *

pre_mod = "valkka.api2.valkkafs: "

# https://en.wikipedia.org/wiki/GUID_Partition_Table#Partition_type_GUIDs
guid_linux_swap="0657FD6D-A4AB-43C4-84E5-0933C84B4F4F"


def findBlockDevices():
    """Scans all block devices for Linux swap partitions, returns key-value pairs like this:
    
    {'626C5523-2979-FD4D-A169-43D818FB0FFE': ('/dev/sda1', 500107862016)}
    
    Key is the uuid of the partition, value is a tuple of the device and its size
    
    To wipe out partition tables from a block device, use:
    
    ::
    
        wipefs -a /dev/devicename
        
    Block devices can be found in
    
    ::
    
        /sys/block/
        
        
    Find block device size in bytes:
        
    ::
    
        blockdev --getsize64 /dev/sdb
        
        
    Show block device id
        
    ::
    
        blkid /dev/sda1
        
    """
    from subprocess import Popen, PIPE
    
    devs={}
    # devs=[]
    block_devices = glob.glob("/sys/block/sd*")
    # block_devices = glob.glob("/sys/block/*")
    for block_device in block_devices:
        devname = os.path.join("/dev", block_device.split("/")[-1]) # e.g. "/dev/sda"
        lis=["sfdisk", devname, "-J"]
        p = Popen(lis, stdout=PIPE, stderr=PIPE)
        st = p.stdout.read().decode("utf-8").strip()
        # print(">"+st+"<")
        if (len(st)>0):
            dic=json.loads(st)
            # print(dic)
            if "partitiontable" in dic:
                ptable = dic["partitiontable"]
                if "partitions" in ptable:
                    for partition in ptable["partitions"]:
                        if (partition["type"]==guid_linux_swap):
                            p = Popen(["blockdev", "--getsize64", devname], stdout=PIPE, stderr=PIPE)
                            st = p.stdout.read().decode("utf-8").strip()
                            devs[partition["uuid"].lower()]=(partition["node"], int(st))
                            """
                                "size"  : int(st),
                                "uuid"  : partition["uuid"]
                                }
                            """
                            # devs.append((partition["uuid"], partition["node"], int(st)))
    return devs


class ValkkaFS:
    """ValkkaFS instance keeps the book for the block file system.
    
    A single instance refers to one block file system and can be shared between one writer and several reader instances, so it is thread safe.
    
    - Instantiate ValkkaFS with the static methods (loadFromDirectory or newFromDirectory) and then pass them to ValkkaFSWriter and ValkkaFSReader instances
    
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
    
    @staticmethod
    def loadFromDirectory(dirname, verbose=False):
        jsonfile    =os.path.join(dirname,"valkkafs.json")
        assert(os.path.exists(dirname))
        assert(os.path.exists(jsonfile))
        f=open(jsonfile, "r")
        dic=json.loads(f.read())
        f.close()
        
        dumpfile        =dic["dumpfile"]
        blockfile       =dic["blockfile"]
        blocksize       =dic["blocksize"]
        n_blocks        =dic["n_blocks"]
        current_block   =dic["current_block"]
        
        assert(os.path.exists(dumpfile))
        assert(os.path.exists(blockfile))
        assert(current_block < n_blocks)
        
        fs = ValkkaFS(
            dumpfile   =dumpfile, 
            blockfile  =blockfile,
            blocksize  =blocksize,
            n_blocks   =n_blocks,
            current_block =current_block,
            jsonfile   =jsonfile,
            verbose    =verbose
            )
        
        fs.reload_()
        return fs
        
        
    @staticmethod
    def newFromDirectory(dirname=None, blocksize=None, n_blocks=None, device_size=None, partition_uuid=None, verbose=False):
        assert(isinstance(dirname, str))
        if (os.path.exists(dirname)):
            pass
        else:
            os.makedirs(dirname)
        
        assert(isinstance(blocksize, int))
        
        blocksize = max(512, (blocksize - blocksize%512)) # blocksize must be a multiple of 512
        
        if (isinstance(n_blocks, int)):
            pass
        else: # if number of blocks is not defined, we need the devicefile size
            assert(isinstance(device_size, int))
            n_blocks = device_size // blocksize

        jsonfile  =os.path.join(dirname,"valkkafs.json")
        blockfile =os.path.join(dirname,"blockfile")
        
        dumpfile = None
        if (partition_uuid==None): # if no partition has been defined, then use default filename "dumpfile" in the directory
            dumpfile=os.path.join(dirname,"dumpfile")
        
        fs = ValkkaFS(
            partition_uuid = partition_uuid,
            dumpfile   =dumpfile,
            blockfile  =blockfile,
            blocksize  =blocksize,
            n_blocks   =n_blocks,
            current_block =0,
            jsonfile   =jsonfile,
            verbose    =verbose
            )
        
        fs.reinit()
        return fs
        
             
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
    
    def __init__(self, **kwargs):
        self.pre = self.__class__.__name__ + " : "
        # checks kwargs agains parameter_defs, attach ok'd parameters to this
        # object as attributes
        parameterInitCheck(ValkkaFS.parameter_defs, kwargs, self)
        
        block_device_dic = findBlockDevices() 
        
        if (self.dumpfile==None): # no dumpfile defined .. so it must be the partition uuid
            assert(isinstance(self.partition_uuid, str))
            assert(self.partition_uuid in block_device_dic)
            self.dumpfile = block_device_dic[self.partition_uuid.lower()][0]
        else:
            assert(isinstance(self.dumpfile, str))
        
        self.core = core.ValkkaFS(self.dumpfile, self.blockfile, self.blocksize, self.n_blocks, False) # dumpfile, blockfile, blocksize, number of blocks, init blocktable
        self.blocksize = self.core.getBlockSize() # in the case it was adjusted ..
        
        self.blocktable_ = numpy.zeros((self.core.get_n_blocks(), self.core.get_n_cols()),dtype=numpy.int_) # this memory area is accessed by cpp
        # self.blocktable  = numpy.zeros((self.core.get_n_blocks(), self.core.get_n_cols()),dtype=numpy.int_) # copy of the cpp blocktable
        # self.getBlockTable()
        self.core.setBlockCallback(self.new_block_cb) # callback when a new block is registered

    def new_block_cb(self, inp):
        """inp is either an error string or an int
        """
        if (self.verbose):
            print(self.pre, "new_block_cb:", inp)
        if isinstance(inp, int):
            self.current_block=inp
            self.writeJson()
        elif isinstance(inp, str):
            print("ValkkaFS: new_block_cb: got message:", inp)
        
    def getBlockTable(self):
        self.core.setArrayCall(self.blocktable_) # copy data from cpp to python (this is thread safe)
        # self.blocktable[:,:] = self.blocktable_[:,:] # copy the data .. self.blocktable can be accessed without fear of being overwritten during accesss
        # return self.blocktable
        return self.blocktable_
        
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
        # verbose=True
        verbose=False
        self.core.clearTable()
        self.core.dumpTable()
        self.writeJson()
        # for regular, files, write them through
        # for block devices, just write the stripes
        if (self.dumpfile.find("/dev/")>-1): # this is a dedicated device
            print("ValkkaFS: striping device")
            self.core.clearDevice(False, verbose) # just write long int zeros in the beginning of each block
            print("ValkkaFS: striped device")
        else:
            print("ValkkaFS: clearing device")
            self.core.clearDevice(True, verbose) # write-through the file
            print("ValkkaFS: cleared device")
        
    def reload_(self):
        self.core.readTable()
    
    
    def getTimeRange(self, blocktable = None):
        if not blocktable:
            blocktable = self.getBlockTable()
        return (blocktable[:,0].min(), blocktable[:,1].max())
        
    
    def getInd(self, times, blocktable = None):
        """Times is a tuple of mstimestamps.  Returns blocks between mstimestamps
        
        This implementation of searching the blocks might seem overly complicated ..
        .. but its not: there are many things to take into account, for example, the fact
        that block numbers are wrapped over device boundaries
        """
        
        # verbose=True
        verbose=False
        
        if not blocktable:
            blocktable = self.getBlockTable()
        
        t0 = times[0]
        t1 = times[1]

        if (t0>t1):
            return []
            
        # 1: max among keyframes
        # 2: min among all frames
            
        ftab = blocktable[:,0]                 # FTAB: max values  
        filt = blocktable[:,0]>0               # we're only interested in blocks that are used (i.e. > 0)
        
        # ***** FROM LOWER LIMIT TO INF ****
        
        aux = ftab[(ftab<=t0)*filt]            # All blocks before the requested keyframe (including a block that has the keyframe)
        """
        We dont want timestamps that are >= t0, but the last one in the list of values with t>=0 
        
        t0 = 10.5
        t1 = 15.5
        
        15 16 17 7 8 9 10 11 12 13 14
                 . . . .  
                       |
                     take this (max) = tt
        """
    
        if (verbose): print("BlockTable: aux 1=",aux)
        
        if (aux.shape[0]>0):
            tt=aux[aux==aux.max()][0]          # target time found ok
        else:
            aux=ftab[(ftab>=t0)*filt]          # the next best option.. 
            """
            t0 = 10.5
            
            15 16 17 0 0  11 12 13 14
            .  .  .       .  .  .  .
                          |
                          take this (min) = tt
            """
            if (aux.shape[0]==0):             # there is nothing corresponding to t0 !
                return []
            else:
                if (verbose): print("BlockTable: aux 1b=",aux)
                tt=aux.min()
        
        if (verbose): print("BlockTable: target time 1=",tt)
        
        eka = (ftab>=tt)*filt                  # all times greater or equal to final target time tt
        """
        eka:
        
        15 16 17 7 8 9 10 11 12 13 14
        .  .  .         .  .  .  .  . 
        """
        
        if (verbose): print("BlockTable: eka=",eka)
        
        
        # **** FROM -INF TO UPPER LIMIT ****
        
        
        aux=ftab[(ftab>t1)*filt]             # ("target time 2" =tt) is just above t1, i.e. the first timestamp in the blocktable > t1
        """
        We don't want timestamps <= t1, but the smallest in the list of values > t1
        
        
        t0 = 10.5
        t1 = 15.5
        
        15 16 17 7 8 9 10 11 12 13 14
           .  .                     
           | 
           take this == tt
        """
        
        if (aux.shape[0]==0):                # so there's nothing greater than t1
            aux=ftab[(ftab>=t1)*filt]        # .. equal to, maybe?
                    
        if (aux.shape[0]==0):                # nothing, nothing
            toka=eka
        else:
            tt=aux[aux==aux.min()][0]        # target time found
            toka=(ftab<=tt)*filt             # all times less or equal to target times (compared to FTAB: max values)
            """
            15 16 17 7 8 9 10 11 12 13 14
            .  .     . . .  .  .  .  .  .
            """
    

        if (verbose): print("BlockTable: target time 2=",tt)

        if (verbose): print("BlockTable: toka=",toka)
        
        # *** COMBINE (LOWER, INF) and (-INF, UPPER) ***
        inds=numpy.where(eka*toka)[0]        # AND
        
        order=blocktable[inds,0].argsort()   # put the block indices into time order
        if (verbose): print("BlockTable: order:",order)
        if (len(order)>0):
            t0=blocktable[order[0],0]
        else:
            t0=None
            
        return inds[order]                   # return sorted indices
        

    def getIndNeigh(self, n=1, time=0, blocktable = None):
        verbose=False
        # verbose=True
        
        if not blocktable:
            blocktable = self.getBlockTable()
        
        ftab=blocktable[:,0]                                # row index 1 = max timestamp of all devices in the block 
        
        inds=numpy.where( ( (ftab<time) & (ftab>0) ) )[0]   # => ftab indices
        if (verbose): print("BlockTable: inds:",inds)
        # print(ftab[inds])
        order=ftab[inds].argsort()                          # ordering array that puts those indices into time order ..
        if (verbose): print("BlockTable: order:",order)
        inds=inds[order]                                    # sorted indices ..
        # if (verbose): print("Sorted blocks:",ftab[inds])
        if (verbose): print("BlockTable: sorted indices",inds)
        inds=numpy.flipud(inds)                             # .. in descending order
        if (verbose): print("BlockTable: indices, des",inds)
        inds=inds[0:min(len(inds),n)]                       # take nearest neighbour to target block.. and the target block (its the first index .. not necessarily!)
        finalinds=inds.copy()
        if (verbose): print("BlockTable: final indices",finalinds,"\n")
        
        inds=numpy.where(ftab>=time)[0]                     # => ftab indices
        if (verbose): print("BlockTable: inds2:",inds)
        order=ftab[inds].argsort()                          # ordering array that puts those indices into time order ..
        if (verbose): print("BlockTable: order2:",order)
        inds=inds[order]                                    # sorted indices ..
        if (verbose): print("BlockTable: sorted indices2:",inds)
        inds=inds[0:min(len(inds),n+1)]                     # take nearest neighbour to target block..
        if (verbose): print("BlockTable: final indices2",inds,"\n")
        finalinds=numpy.hstack((finalinds,inds))
        if (verbose): print("BlockTable: ** final indices",inds,"\n")
        
        finalinds.sort()
        return finalinds
    

    
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



 
