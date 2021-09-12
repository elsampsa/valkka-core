import time, logging
from valkka import core # ValkkaFSWriterThread, ValkkaFSReaderThread, FileCacheThread # api level 1
from valkka.fs.group import FSGroup
from valkka.fs.multi import ValkkaMultiFS
from valkka.fs.single import ValkkaSingleFS
from valkka.api2.tools import getLogger, setLogger


class ValkkaFSManager:
    """Manages a group of ValkkaFS instances

    - Common core.FileCacheThread where all frames are cached
    - For each ValkkaFS, create a core.ValkkaReaderThread and core.ValkkaFSWriterThread

    :param valkafs_list: A list of api level 2 ValkkaFS objects

    ::

        core.ValkkaFSReaderThread
        core.ValkkaFSReaderThread  ---> core.FileCacheThread
        ...


    API example:


    ::
        # manager associates certain api2.ValkkaFS to a unique
        # core.ValkkaFSReaderThread and core.ValkkaFSWriterThread
        # It also creates a single core.FileCacheThread thread 

        manager = ValkkaFSGroupManager(valkkafs_list)
        
        # read/write id <-> slot association,
        # valkkafs <-> slot association
        manager.setSlotId(valkkafs, 1, 1001)

        # get associated core.ValkkaFSWriterThread input filter:
        in_filter = manager.getInputfilter(valkkafs)

        # write from certain valkkafs (from associated slot) into out_filter
        manager.setOutputFilter(valkkafs, out_filter)

    """
    def __init__(self, valkkafs_list: list, name = "manager"):
        self.logger = getLogger(self.__class__.__name__)
        setLogger(self.logger, logging.DEBUG)
        self.name = name
        self.cacherthread = core.FileCacheThread(self.name)
        self.fsgroups = []
        self.fsgroup_by_valkkafs = {}
        self.timerange_by_valkkafs = {}
        self.new_block_cb = None
        self.timerange = (0, 0) # should be queried by, say, a widget
        for valkkafs in valkkafs_list:
            fsgroup = FSGroup(
                    valkkafs,
                    core.ValkkaFSReaderThread("reader", 
                        valkkafs.core, 
                        self.cacherthread.getFrameFilter()
                        ),
                    core.ValkkaFSWriterThread("writer", 
                        valkkafs.core
                        )
                )
            fsgroup.setBlockCallback(self.new_block_cb__)
            self.fsgroups.append(fsgroup)
            self.fsgroup_by_valkkafs[valkkafs] = fsgroup
            self.timerange_by_valkkafs[valkkafs] = (0,0)
            
        self.cacherthread.setPyCallback(self.timeCallback__)
        self.cacherthread.setPyCallback2(self.timeLimitsCallback__)
        self.started = False


    def __del__(self):
        self.close()


    def getInputFilter(self, valkkafs):
        fsgroup = self.fsgroup_by_valkkafs[valkkafs]
        return fsgroup.getInputFilter()
            

    def getTimeRange(self):
        """API end-point for widgets, etc.
        """
        return self.timerange


    def getTimeRangeByValkkaFS(self, valkkafs):
        """API end-point for widgets, etc.
        """
        return self.timerange_by_valkkafs[valkkafs]


    def start(self):
        self.cacherthread.startCall()
        for fsgroup in self.fsgroups:
            fsgroup.start()
        self.readBlockTablesIf()
        

    def close(self):
        if not self.started:
            return
        for fsgroup in self.fsgroups:
            fsgroup.requestStop()
        for fsgroup in self.fsgroups:
            fsgroup.waitStop()
        self.cacherthread.stopCall()
        self.started = False


    def setTimeCallback(self, timecallback: callable):
        """Continue the callback chain, originating from FileCacheThread::run
        
        Callback carries current seek/playtime
        """
        self.timecallback = timecallback
        
        
    def setTimeLimitsCallback(self, timelimitscallback: callable):
        """Continue the callback chain, originating from FileCacheThread::switchcache

        Callback carries the timelimits of currently cached frames (as a tuple)
        """
        self.timelimitscallback = timelimitscallback
        
        
    def setBlockCallback(self, func):
        """

        func should comply with this:

        ::

            func(
                    valkkafs = {can be None}
                    timerange_local = {tuple, can be None}
                    timerange_global = tuple
                )
        """
        self.new_block_cb = func


    def map_(self, valkkafs=None, framefilter=None,
            write_slot=None, read_slot=None, _id=None):
        """Mapping valkkafs (i.e. file) => framefilter

        User needs to make up an id.  It can be identical
        to the slot number
        """
        fsgroup = self.fsgroup_by_valkkafs[valkkafs]
        fsgroup.map_(
            read_slot = read_slot,
            write_slot = write_slot,
            _id = _id,
            framefilter = framefilter
        )
        ctx = fsgroup.getFileStreamContext(_id)
        self.cacherthread.registerStreamCall(ctx) # slot => framefilter mapping by core.FileCacheThread
        
    def unmap(self, valkkafs=None, _id=None):
        fsgroup = self.fsgroup_by_valkkafs[valkkafs]
        ctx = fsgroup.getFileStreamContext(_id)
        fsgroup.unmap(_id)
        self.cacherthread.deregisterStreamCall(ctx)
        
    def updateTimeRange__(self):
        """update the global timerange from all blocktables
        """
        mintime = 9999999999
        maxtime = -1
        for fsgroup in self.fsgroups:
            if fsgroup.timerange is None: # empty BT
                continue
            mintime = min(fsgroup.timerange[0], mintime)
            maxtime = max(fsgroup.timerange[1], maxtime)
            self.timerange_by_valkkafs[fsgroup.valkkafs] = fsgroup.timerange
        self.timerange = (mintime, maxtime)


    def readBlockTableIf(self, fsgroup):
        """Re-read a single bloctable, update timerange
        """
        # fsgroup = self.fsgroup_by_valkkafs[valkkafs]
        assert(fsgroup in self.fsgroups)
        fsgroup.readBlockTableIf()
        self.updateTimeRange__()


    def readBlockTablesIf(self):
        for fsgroup in self.fsgroups:
            fsgroup.readBlockTableIf()
        self.updateTimeRange__()


    def reqBlocksIf(self):
        for fsgroup in self.fsgroups:
            fsgroup.reqBlocks()


    def new_block_cb__(self, fsgroup):
        """Called when block information in valkkafs has been updated

        Information to propagate to GUI, could be, for example:
        - valkkafs instance
        - timelimit of the valkkafs instance in question
        - global timelimit of all valkkafs instances
        """
        self.readBlockTableIf(fsgroup)
        if self.new_block_cb is not None:
            self.new_block_cb(
                valkkafs = fsgroup.valkkafs,
                timerange_local = fsgroup.timerange,
                timerange_global = self.timerange
            )

        
    def timeCallback__(self, mstime: int):
        """This is called from cpp on regular time 
        intervals from FileCacheThread
            
        :param mstime:  play/seek millisecond timestamp
            
        Called from cpp side:

        ::

            cachestream.cpp
                FileCacheThread
                    run
                        calls pyfunc (this method)
        """
        try:
            """
            Handle under / overflows:
            
            - If the stream goes over global timelimits, stop it
            - .. same for underflow
            - If goes under / over currently available blocks, order more
            
            - using try / except blocks we can see the error message even when this is called from cpp
            """
            # refresh all necessary valkkafs
            # blocktables
            self.readBlockTablesIf()

            if mstime <= 0: # time not set
                self.logger.debug("timeCallback__ : no time set")
                # print("timeCallback__ : no time set")
            else:
                """
                if not self.timeOK(mstime) and self.playing:
                    self.logger.warning("timeCallback__ : no such time in blocktable %i", (mstime))
                    self.logger.warning("timeCallback__ : stop stream")
                    self.stop()
                    return
                """
                # request blocks around certain 
                # millisecond timestamp from all
                # necessary valkkafs
                self.reqBlocksIf(mstime)
                """
                # this logic now in FSGroup.reqBlocksIf
                #
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
                    return
                """
            
            self.currentmstime = mstime
            if self.timecallback is not None:
                try:
                    self.timecallback(mstime)
                except Exception as e:
                    self.logger.warning("timeCallback__ : your callback failed with '%s'", str(e))
                    
        except Exception as e:
            # raise(e)
            self.logger.debug("timeCallback__ failed with '%s'" % (str(e)))
    

    def timeLimitsCallback__(self, tup):
        """Called from cpp side, see:

        ::

            cachestream.cpp
                FileCacheThread
                    switchCache
                        calls pyfunc2 (this method)

        TODO: analyzer cpp-python-cpp callchain

        Carries a tuple with the timelimits of the currently cached frames
        """
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
        


def managertest():
    # create a dummy ValkkaFS
    valkkafsclass = ValkkaMultiFS
    # valkkafsclass = ValkkaSingleFS

    valkkafs_1 = valkkafsclass.newFromDirectory(
        dirname = "./vfs1",
        blocksize = 1024*1024,
        n_blocks = 5,
        verbose = True)
    valkkafs_2 = valkkafsclass.newFromDirectory(
        dirname = "./vfs2",
        blocksize = 1024*1024,
        n_blocks = 5,
        verbose = True)
    manager = ValkkaFSManager([valkkafs_1, valkkafs_2])
    # manager creates internally a FileCacheThread
    manager.start() # starts all threads

    # these would go into AVThreads..
    out_filter_1 = core.InfoFrameFilter("info1")
    out_filter_2 = core.InfoFrameFilter("info2")

    # reader & writer associated with 
    # valkkafs_1 map internally slot 1 to id 1001 on disk
    # output of cacherthread goes into out_filter_1:
    manager.map_(
        valkkafs = valkkafs_1,
        framefilter = out_filter_1,
        write_slot = 1,
        read_slot =1,
        _id = 1001)
    # etc..
    manager.map_(
        valkkafs = valkkafs_2,
        framefilter = out_filter_2,
        write_slot = 2,
        read_slot = 2,
        _id = 2002)
    # remember that is possible to write 
    # multiple streams into the same ValkkaFS
    manager.unmap(valkkafs=valkkafs_1, _id=1001)
    manager.unmap(valkkafs=valkkafs_2, _id=2002)


if __name__ == "__main__":
    managertest()

