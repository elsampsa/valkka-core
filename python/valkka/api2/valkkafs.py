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
@version 0.15.0 

@brief   api level 1 => api level 2 encapsulation for ValkkaFS and ValkkaFSThreads
"""
import time
import json
import os
import glob
import numpy
import logging
from valkka import core
from valkka.api2.tools import *
from valkka.api2.exceptions import ValkkaFSLoadError
import datetime
import traceback

pre_mod = "valkka.api2.valkkafs: "

# https://en.wikipedia.org/wiki/GUID_Partition_Table#Partition_type_GUIDs
guid_linux_swap="0657FD6D-A4AB-43C4-84E5-0933C84B4F4F" # GPT partition table
mbr_linux_swap="82" # MBR partition table


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
    block_devices += glob.glob("/dev/nvme*")
    # block_devices = glob.glob("/sys/block/*")
    for block_device in block_devices:
        # print("block_device", block_device)
        devname = os.path.join("/dev", block_device.split("/")[-1]) # e.g. "/dev/sda"
        lis=["sfdisk", devname, "-J"]
        # print("lis>", lis)
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
                        if (partition["type"]==guid_linux_swap): # GTP partition table 
                            p = Popen(["blockdev", "--getsize64", devname], stdout=PIPE, stderr=PIPE)
                            st = p.stdout.read().decode("utf-8").strip()
                            devs[partition["uuid"].lower()]=(partition["node"], int(st))
                            """
                                "size"  : int(st),
                                "uuid"  : partition["uuid"]
                                }
                            """
                            # devs.append((partition["uuid"], partition["node"], int(st)))
                        """
                        elif (partition["type"]==mbr_linux_swap): # MBR partition table
                            p = Popen(["blockdev", "--getsize64", devname], stdout=PIPE, stderr=PIPE)
                            st = p.stdout.read().decode("utf-8").strip()
                            devs[partition["node"].lower()]=(partition["node"], int(st))
                        """
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
        
        fs = ValkkaFS(
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
        
        
    @staticmethod
    def newFromDirectory(dirname=None, blocksize=None, n_blocks=None, device_size=None, partition_uuid=None, verbose=False):
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
        
        fs = ValkkaFS(
            partition_uuid = partition_uuid,
            dumpfile   = dumpfile,
            blockfile  = blockfile,
            blocksize  = blocksize,
            n_blocks   = n_blocks,
            current_block = 0,
            jsonfile   = jsonfile,
            verbose    = verbose
            )
        
        fs.reinit() # striped the device
        return fs
        
        
    @staticmethod
    def checkDirectory(dirname):
        jsonfile  =os.path.join(dirname,"valkkafs.json")
        assert(os.path.exists(dirname))
        assert(os.path.exists(jsonfile))
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
    
    def __init__(self, **kwargs):
        """If partition_uuid defined, the dumpfile is set by ctor to the device defined by partition_uuid
        """
        self.pre = self.__class__.__name__ + " : "
        # checks kwargs agains parameter_defs, attach ok'd parameters to this
        # object as attributes
        parameterInitCheck(ValkkaFS.parameter_defs, kwargs, self)
        
        self.block_cb = None # a custom callback in the callback chain when a new block is ready
        
        block_device_dic = findBlockDevices() 
        
        if (self.dumpfile==None): # no dumpfile defined .. so it must be the partition uuid
            assert(isinstance(self.partition_uuid, str))
            assert(self.partition_uuid in block_device_dic)
            self.dumpfile = block_device_dic[self.partition_uuid.lower()][0]
        else:
            assert(isinstance(self.dumpfile, str))
        
        print("ValkkaFS: dumpfile = %s" % (str(self.dumpfile)))
        
        self.core = core.ValkkaFS(self.dumpfile, self.blockfile, self.blocksize, self.n_blocks, False) # dumpfile, blockfile, blocksize, number of blocks, init blocktable
        
        self.blocksize = self.core.getBlockSize() # in the case it was adjusted ..
        
        self.blocktable_ = numpy.zeros((self.core.get_n_blocks(), self.core.get_n_cols()),dtype=numpy.int_) # this memory area is accessed by cpp
        # self.blocktable  = numpy.zeros((self.core.get_n_blocks(), self.core.get_n_cols()),dtype=numpy.int_) # copy of the cpp blocktable
        # self.getBlockTable()
        self.core.setBlockCallback(self.new_block_cb__) # callback when a new block is registered

        print("ValkkaFS: resuming writing at block", self.current_block)

        self.core.setCurrentBlock(self.current_block)

        # # attach an analysis tool
        # self.analyzer = core.ValkkaFSTool(self.core)
        
        self.verbose = True


    def is_same(self, **dic):
        """Compare this ValkkaFS to given parameters

        Use parameters that are not modified by the ctor
        """
        print("partition_uuid:",self.partition_uuid,dic["partition_uuid"])
        print("blocksize     :",self.blocksize, dic["blocksize"])
        print("n_blocks      :", self.n_blocks, dic["n_blocks"])

        return (\
        (self.partition_uuid  == dic["partition_uuid"]) and \
        (self.blocksize       == dic["blocksize"]) and \
        (self.n_blocks        == dic["n_blocks"]) \
        )




    def new_block_cb__(self, propagate, par):
        """input tuple elements:
        
        boolean    : should the callback be propagated or not
        int / str  : int = block number, str = error message
        
        """
        
        #propagate = tup[0]
        #par = tup[1]
        
        try:
            if (self.verbose):
                print(self.pre, "new_block_cb__:", propagate, par)
            if isinstance(par, int):
                self.current_block = par
                print("ValkkaFS: new_block_cb__: block:", par)
                self.writeJson()
                print("ValkkaFS: new_block_cb__: wrote json")
            elif isinstance(par, str):
                print("ValkkaFS: new_block_cb__: got message:", par)
                
            if (self.block_cb is not None) and propagate:
                print(self.pre, "new_block_cb: subcallback")
                self.block_cb()
                
            print(self.pre, "new_block_cb: bye")
                
        except Exception as e:
            print("ValkkaFS: failed with '%s'" % (str(e)))
            
        
        
        
    def setBlockCallback(self, cb):
        self.block_cb = cb
        
        
    def getBlockTable(self):
        print("ValkkaFS: getBlockTable")
        # traceback.print_stack()
        self.core.setArrayCall(self.blocktable_) # copy data from cpp to python (this is thread safe)
        # self.blocktable[:,:] = self.blocktable_[:,:] # copy the data .. self.blocktable can be accessed without fear of being overwritten during accesss
        # return self.blocktable
        print("ValkkaFS: getBlockTable: exit")
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
        # TODO: remove dumpfile
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
        if isinstance(blocktable,None.__class__):
            blocktable = self.getBlockTable()
        nonzero = blocktable[:,0] > 0
        
        if nonzero.sum() < 1: # empty blocktable
            return ()
            
        return (int(blocktable[nonzero,0].min()), int(blocktable[nonzero,1].max()))
        
    
    def getInd(self, times, blocktable = None):
        """Times is a tuple of mstimestamps.  Returns blocks between mstimestamps
        
        This implementation of searching the blocks might seem overly complicated ..
        .. but its not: there are many things to take into account, for example, the fact
        that block numbers are wrapped over device boundaries
        """
        
        # verbose=True
        verbose=False
        
        if isinstance(blocktable,None.__class__):
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
            
        final = inds[order]                 # return sorted indices
            
        return final.tolist()
        

    def getIndNeigh(self, n=1, time=0, blocktable = None):
        verbose=False
        # verbose=True
        
        if isinstance(blocktable,None.__class__):
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
        if (verbose): print("BlockTable: ** final indices",finalinds,"\n")
        
        order=ftab[finalinds].argsort()                          # final blocks should be in time order
        finalinds = finalinds[order]
        # finalinds.sort()
        return finalinds.tolist()
    

    
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



def formatMstimestamp(mstime):
    t = datetime.datetime.fromtimestamp(mstime/1000)
    return "%s:%s" % (t.minute, t.second)
    


"""
- ValkkaFS
- ValkkaFSReaderThread
- FileCacheThread
                                            (+---->   AVThread)
                                    SOURCE     |                    
valkkafsreaderthread --> filecachethread ----+---->   AVThread  ----> Forks, etc. ----> OpenGLThread

id => slot               filestreamcontext:
                            slot => framefilter

how about .. ?

SOURCE ==> ManagerFilterChain

ManagedFilterChain.setSource(filterchain)

- camera database: id numbers
- create managedfilterchain for every file stream

LiveManagedFilterChain (source defined within the class)
USBManagedFilterChain (same ..)
FileManagedFilterChain .. or just
ManagedFilterChain

For every camera, generate a ManagedFilterChain

ManagedFilterChain.getInputFrameFilter(framefilter)
ValkkaFSManager.setOutput(id, slot, framefilter)
"""
class ValkkaFSManager:
    """Takes as an input a (api level 2) ValkkaFS
    
    Instantiates & encapsulates api 1 level objects:
    
    - ValkkaFSWriterThread : Writes stream to disk
    - ValkkaFSReaderThread : Reads stream from disk & passes them to FileCacheThread
    - FileCacheThread : Caches frames & passes them at play speed downstream
    
        - ValkkaFSManager attaches FileCacheThread callbacks for various events
    
    All those objects operate on a common ValkkaFS object (ValkkaFS manages the blocktable)
    
    Frame writing API:
    
    ::
    
        setInput(_id, slot)   :  declare slot to id mapping for writing frames
        TODO: remove mapping
        getFrameFilter()      :  returns framefilter for writing frames
        
    
    """
    
    
    
    timetolerance = 2000 # if frames are missing at this distance or further, request for more blocks
    timediff = 5 # blocktable can be inquired max this frequency (secs)
    
    def __init__(self, valkkafs: ValkkaFS, write = True, read = True, cache = True):
        """ValkkaFSReaderThread --> FileCacheThread
        """
        self.pre = __name__ + "." + self.__class__.__name__
        self.logger = getLogger(self.pre)
        setLogger(self.pre, logging.DEBUG)
        # self.logger.setLevel(logging.INFO)
        # self.logger.setLevel(logging.WARNING)
        
        self.valkkafs = valkkafs
        
        self.timecallback = None
        self.timelimitscallback = None
        
        self.timerange = () # filesystem timerange.  empty tuple implies no frames
        
        self.readBlockTable()
        
        self.cacherthread = core.FileCacheThread("cacher")
        self.readerthread = core.ValkkaFSReaderThread("reader", self.valkkafs.core, self.cacherthread.getFrameFilter())
        self.writerthread = core.ValkkaFSWriterThread("writer", self.valkkafs.core)
        
        self.currentmstime = None # None means there's no reference point (no seek has succeeded so far)
        self.current_blocks = []
        self.current_timerange = (0,0) # time limits of current blocks
        
        # callback coming from cpp, informing about the current time
        self.cacherthread.setPyCallback(self.timeCallback__)
        self.cacherthread.setPyCallback2(self.timeLimitsCallback__)
        
        # use these for debugging
        if cache:
            self.cacherthread.startCall()
        if read:
            self.readerthread.startCall()
        if write:
            self.writerthread.startCall()
            
        self.active = True
        self.playing = False
        
        
    def hasFrames(self):
        return len(self.timerange) > 0
        
        
    def setTimeCallback(self, timecallback: callable):
        self.timecallback = timecallback
        
        
    def setTimeLimitsCallback(self, timelimitscallback: callable):
        self.timelimitscallback = timelimitscallback
        
        
    def setBlockCallback(self, cb):
        self.valkkafs.setBlockCallback(cb)
        
        
    def timeCallback__(self, mstime: int):
        # return # debug
        try:
            """This is called from cpp on regular time intervals
            
            :param mstime:  millisecond timestamp
            
            Handle under / overflows:
            
            - If the stream goes over global timelimits, stop it
            - .. same for underflow
            - If goes under / over currently available blocks, order more
            
            - using try / except blocks we can see the error message even when this is called from cpp
            """
            
            # self.logger.debug("timeCallback__ : %i == %s", mstime, formatMstimestamp(mstime))
            
            if mstime <= 0: # time not set
                # self.logger.debug("timeCallback__ : no time set")
                return
            
            self.readBlockTableIf()
            if not self.timeOK(mstime) and self.playing:
                self.logger.warning("timeCallback__ : no such time in blocktable %i", (mstime))
                self.logger.warning("timeCallback__ : stop stream")
                # self.playing = False
                self.stop()
                return
            
            # self.logger.debug("timeCallback__ : distance to current block edge is %i", mstime-self.current_timerange[1])
            
            if self.current_timerange[1] - mstime < self.timetolerance: # time to request more blocks
                if (mstime >= (self.timerange[0] + 2000) and mstime <= (self.timerange[1] - 2000)): 
                    # request blocks only if the times are in blocktable
                    # the requested time must also be a bit away from the filesystem limit (self.timerange[1])
                    self.logger.info("timeCallback__ : will request more blocks: time: %s, timerange: %s", mstime, self.timerange)
                    ok = self.reqBlocks(mstime)
                    if not ok:
                        self.logger.warning("timeCallback__ : requesting blocks failed")
               
            if not self.currentTimeOK(mstime):
                self.logger.info("timeCallback__ : no frames for time %i", (mstime))
                # TODO
                return
            
            # TODO: time near block limits: request for more frames if possible
            # TODO: does the framecache switching go smoothly in play?
            
            # self.logger.debug("timeCallback__ : time ok : %i" % (mstime))
            self.currentmstime = mstime
            
            if self.timecallback is not None:
                try:
                    self.timecallback(mstime)
                except Exception as e:
                    self.logger.warning("timeCallback__ : your callback failed with '%s'", str(e))
                    
        except Exception as e:
            raise(e)
            self.logger.warning("timeCallback__ failed with '%s'" % (str(e)))
            
            
    def timeLimitsCallback__(self, tup):
        try:
            self.logger.debug("timeLimitsCallback__ : %s", str(tup))
            self.logger.debug("timeLimitsCallback__ : %s -> %s", formatMstimestamp(tup[0]), formatMstimestamp(tup[1]))
            if self.timelimitscallback is not None:
                self.timelimitscallback(tup)
        except Exception as e:
            print("timeLimitsCallback__ failed with '%s'" % (str(e)))
        
        
    def getCurrentTime(self):
        return self.currentmstime
   
   
    def timeOK(self, mstimestamp):
        """Time in the blocktable range?
        """
        if self.hasFrames() == False:
            return False
        if (mstimestamp < self.timerange[0]) or (mstimestamp > self.timerange[1]):
            self.logger.info("timeOK : invalid time %i range is %i %i", 
                                mstimestamp, self.timerange[0], self.timerange[1])
            self.logger.info("timeOK : %i %i", 
                                mstimestamp-self.timerange[0], mstimestamp-self.timerange[1])
            
            return False
        return True
    
    
    def currentTimeOK(self, mstimestamp):
        """Time in the currently loaded blocks range?
        """
        if (mstimestamp < self.current_timerange[0]) or (mstimestamp > self.current_timerange[1]):
            self.logger.info("currentTimeOK : invalid time %i range is %i %i", 
                                mstimestamp, self.current_timerange[0], self.current_timerange[1])
            return False
        return True
    

    def clearTime(self):
        self.cacherthread.clearCall()


    def setOutput(self, _id, slot, framefilter):
        """Set id => slot mapping.  Send to defined framefilter
        """
        self.readerthread.setSlotIdCall(slot, _id) # id => slot
        ctx = core.FileStreamContext(slot, framefilter) # slot => framefilter
        self.cacherthread.registerStreamCall(ctx)
        return ctx
    
    
    def clearOutput(self, ctx):
        self.cacherthread.deregisterStreamCall(ctx)
    
    
    def setOutput_(self, _id, slot):
        """Just set id => slot mapping.  For debugging
        """
        self.readerthread.setSlotIdCall(slot, _id) # id => slot
        
    
    def setInput(self, _id, slot):
        """Map incoming frames from slot to id number
        """
        self.writerthread.setSlotIdCall(slot, _id)
    
    
    def clearInput(self, slot):
        """Unmap incoming frames from slot to id number
        """
        self.writerthread.unSetSlotIdCall(slot)
    
    
    def getFrameFilter(self):
        """Push frames here
        """
        return self.writerthread.getFrameFilter()
    

    def readBlockTable(self):
        """Create a copy of the blocktable
        """
        self.blocktable = self.valkkafs.getBlockTable()
        # self.logger.debug("readBlockTable: %s", self.blocktable)
        self.timerange = self.valkkafs.getTimeRange(self.blocktable)
        self.checktime = time.time()
        
    
    def readBlockTableIf(self):
        """Create a copy of the blocktable, but only if a certain time has passed
        """
        if (time.time() - self.checktime) >= self.timediff:
            self.readBlockTable()


    def play(self):
        if self.currentmstime is None:
            return False
        self.logger.debug("play")
        self.cacherthread.playStreamsCall()
        self.playing = True
        return True
    
    
    def stop(self):
        if self.playing:
            # traceback.print_stack()
            self.logger.debug("stop")
            self.cacherthread.stopStreamsCall()
            self.playing = False
        
    
    def seek(self, mstimestamp):
        """Before performing seek, check the blocktable
        
        - Returns True if the seek could be done, False otherwise
        - Requests frames always from ValkkaFSReaderThread (with reqBlocks), even if there are already frames at this timestamp
        
        """
        assert(isinstance(mstimestamp, int))
        # self.stop() # seekStreamsCall stops
        self.readBlockTableIf()
        if not self.timeOK(mstimestamp):
            return False
        # there's stream for that time, so proceed
        self.logger.info("seek : proceeds with %i", mstimestamp)
        self.cacherthread.seekStreamsCall(mstimestamp, True) # note that clear flag is True : clear the seek.
        ok = self.reqBlocks(mstimestamp)
        if not ok: self.logger.warning("seek : could not get blocks")
            
            
    def smartSeek(self, mstimestamp):
        """Like seek, but does not request frames if there are already frames in this timerange
        """
        self.readBlockTableIf()
        self.logger.debug("smartSeek : current timerange: %s", self.current_timerange)
        self.logger.debug("smartSeek : timetolerance: %s", self.timetolerance)
        if self.hasFrames() == False:
            return
        
        lower = self.current_timerange[0] + self.timetolerance
        upper = self.current_timerange[1] - self.timetolerance
        
        self.logger.debug("smartSeek: mstimestamp %s", mstimestamp)
        self.logger.debug("smartSeek: lower limit %s", lower)
        self.logger.debug("smartSeek: upper limit %s", upper)
        
        if (mstimestamp > lower) and (mstimestamp < upper): # no need to request new blocks
            # self.stop()
            self.logger.debug("smartSeek : just set target time")
            self.cacherthread.seekStreamsCall(mstimestamp, False) # simply sets the target (note that clear is set to False)
        else:
            self.logger.debug("smartSeek : request frames")
            self.seek(mstimestamp)
        
        
    def reqBlocks(self, mstimestamp):
        block_list = self.valkkafs.getIndNeigh(n=1, time=mstimestamp, blocktable = self.blocktable)
        # blocks are now in timeorder
        if len(block_list) > 0:
            self.current_blocks = block_list
            tmp = self.blocktable[block_list,:]
            self.current_timerange = (
                tmp[:,0].min(), # max of the first column (key frames) 
                tmp[:,1].max() # min of the second column (any frames)
                )
            self.logger.debug("reqBlocks : current timerange now %s", self.current_timerange)
            self.logger.debug("reqBlocks : requesting blocks %s %s", block_list.__class__.__name__, str(block_list))
            self.readerthread.pullBlocksPyCall(block_list) # this send frames from readerthread -> cacherthread
            return True
        else:
            return False
        
        
    def getTimeRange(self):
        self.readBlockTableIf()
        return self.timerange
        
        
    def close(self):
        if not self.active:
            return
        self.logger.debug("close: stopping threads")
        """
        self.writerthread.stopCall()
        self.readerthread.stopCall()
        self.cacherthread.stopCall()
        self.active = False
        """
        self.requestClose()
        self.waitClose()
        

    def requestClose(self):
        self.writerthread.requestStopCall()
        self.cacherthread.requestStopCall()
        self.readerthread.requestStopCall()
        
        
    def waitClose(self):
        self.writerthread.waitStopCall()
        self.logger.debug("waitClose: writerthread %s exit", id(self.writerthread))
        self.cacherthread.waitStopCall()
        self.logger.debug("waitClose: cacherthread %s exit", id(self.cacherthread))
        self.readerthread.waitStopCall()
        self.logger.debug("waitClose: readerthread %s exit", id(self.readerthread))
        self.active = False


    def __del__(self):
        self.close()


 
