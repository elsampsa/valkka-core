import time
from valkka import core # ValkkaFSWriterThread, ValkkaFSReaderThread, FileCacheThread # api level 1
from valkka.fs.base import ValkkaFS # api level 2

class FSGroup:
    """
    :param valkkafs: ValkkaFS api level 2 instance
    :param reader: ValkkaFSReaderThread instance.  Not started.  Output framefilter of
    this thread should be connected to the correct FileCacheThread
    :param writer: ValkkaFSWriterThread instance. Not started


    Several streams (with unique slot numbers) can be diverted into
    the ValkkaFSWriterThread input filter (they all go into the same ValkkaFS / file).

    When reading, all frames come from the same ValkkaFSReaderThread's 
    (from the same ValkkaFS / file ) output filter.

    Slots are mapped to unique id numbers in the ValkkaFS file (as defined
    by the user)

    API example:

    ::

        fsgroup = FSGroup(
            valkkafs, 
            core.ValkkaFSReaderThread("reader", valkkafs.core, cachethread.getFrameFilter()),
            core.ValkkaFSWriterThread("writer", valkkafs.core)
            )

        fsgroup.start()

        fsgroup.setSlotId(1, 1001) # slot, id
        in_filter = fsgroup.getInputFilter() # write incoming frames here
        out_filter = ... # AVThread's input filter, for example
        ctx = fsgroup.setOutputFilter(filter = out_filter, slot = 1) # returns FileStreamContext
        cachethread.registerStreamCall(ctx)
        ...
        ...
        ctx = fsgroup.getFileStreamContext()
        cachethread.deregisterStreamCall(ctx)

        fsgroup.stop()

    """
    timetolerance = 2000 # if frames are missing at this distance or further, 
    # request for more blocks
    timediff = 5 # blocktable can be inquired max this frequency (secs)

    def __init__(self, valkkafs: ValkkaFS = None, 
        reader: core.ValkkaFSReaderThread = None,
        writer: core.ValkkaFSWriterThread = None):
        assert(isinstance(valkkafs, ValkkaFS))
        assert(isinstance(reader, core.ValkkaFSReaderThread))
        assert(isinstance(writer, core.ValkkaFSWriterThread))
        self.valkkafs = valkkafs
        self.reader = reader
        self.writer = writer
        self.started = False
        self.valkkafs.setBlockCallback(self.new_block_cb__)
        self.cb = None # a custom callback
        self.current_blocks = [] # currently cached blocks
        self.timerange = (0,0) # timerange of currently cached blocks
        self.checktime = 0 # when BT was requested for the last time
        self.readBlockTable() # inits timerange & checktime

    def getInputFilter(self):
        return self.writer.getFrameFilter()

    def setSlotId(self, slot, _id):
        """Slot <-> id mapping for writing/reading frames to/from disk
        """
        self.writer.setSlotIdCall(slot, _id)
        self.reader.setSlotIdCall(slot, _id)

    def clearSlotId(self, slot):
        self.writer.unSetSlotIdCall(slot)
        self.reader.unSetSlotIdCall(slot)

    def setOutputFilter(self, filter = None, slot = None):
        self.file_stream_ctx = core.FileStreamContext(slot, filter)
        return self.file_stream_ctx

    def getFileStreamContext(self):
        return self.file_stream_ctx

    def start(self):
        self.reader.startCall()
        self.writer.startCall()
        self.started = True
        
    def stop(self):
        self.reader.stopCall()
        self.writer.stopCall()
        self.started = False

    def requestStop(self):
        self.reader.requestStopCall()
        self.writer.requestStopCall()
        self.started = False

    def waitStop(self):
        self.reader.waitStopCall()
        self.writer.waitStopCall()

    def setBlockCallback(self, cb):
        """Set a callback that is fired when there's a new block/
        the block information is updated

        The callback should accept a single parameter
        """
        self.cb = cb

    def new_block_cb__(self):
        """This is always called at a new block
        """
        if self.cb is not None:
            self.cb(self.valkkafs) # emit the apropriate ValkkaFS with the callback


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


    def reqBlocksIf(self, mstime):
        """Request new blocks

        - Update blocktable, if necessary 
        - Request blocks around mstime if there are some & if the new block numbers
        would be different from the previous ones

        :param mstime: target play/seek time
        """
        #
        # TODO
        # - need some more logic here:
        #   even if the mstime falls within min/max range
        #   if there are not frames within.. 5 minutes of the requested time
        #   then don't bother requesting blocks (i.e. there's a gap)
        #

        # (1) check if mstime falls within the current block timerange
        if (mstime-self.timerange[0]) > self.timetolerance:
            return False
        if (self.timerange[1]-mstime) > self.timetolerance:
            return False
        # .. we're well inside the timelimits
        # otherwise:
        block_list = self.valkkafs.getIndNeigh(
            n=1, 
            time=mstime, 
            blocktable = self.blocktable)
        # block numbers are in time order
        # (2) if blocks would be different than the ones we have currently,
        # request them
        if block_list == self.current_blocks:
            return False
        # blocks ok
        self.current_blocks = block_list
        tmp = self.blocktable[self.current_blocks,:]
        self.current_timerange = (
            tmp[:,0].min(), # max of the first column (key frames) 
            tmp[:,1].max() # min of the second column (any frames)
            )
        self.logger.debug("reqBlocksIf : current timerange now %s", 
            self.current_timerange)
        self.logger.debug("reqBlocksIf : requesting blocks %s", 
            str(self.current_blocks))
        self.reader.pullBlocksPyCall(self.current_blocks) 
        # ..send frames from readerthread -> cacherthread
        return True



class ValkkaFSManager:
    """Manages a group of ValkkaFS instances

    - Common core.FileCacheThread where all frames are cached
    - For each ValkkaFS, create a core.ValkkaReaderThread and core.ValkkaFSWriterThread

    :param valkafs_list: A list of api level 2 ValkkaFS objects

    Filterchain: TODO

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
        self.name = name
        self.cacherthread = core.FileCacheThread(self.name)
        self.fsgroups = []
        self.fsgroup_by_valkkafs = {}
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

        self.cacherthread.setPyCallback(self.timeCallback__)
        self.cacherthread.setPyCallback2(self.timeLimitsCallback__)
        self.started = False


    def __del__(self):
        self.stop()
    

    def start(self):
        self.cacherthread.startCall()
        for fsgroup in self.fsgroups:
            fsgroup.start()

    def stop(self):
        if not self.started:
            return
        for fsgroup in self.fsgroups:
            fsgroup.requestStop()
        for fsgroup in self.fsgroups:
            fsgroup.waitStop()
        self.cacherthread.stopCall()
        self.started = False


    def map_(self, valkkafs=None, framefilter=None,
            slot=None, _id=None):
        """Mapping valkkafs (i.e. file) => framefilter

        User needs to make up an id.  It can be identical
        to the slot number
        """
        fsgroup = self.fsgroup_by_valkkafs[valkkafs] 
        reader = fsgroup.reader
        writer = fsgroup.writer
        reader.setSlotIdCall(slot, _id)
        writer.setSlotIdCall(slot, _id)
        ctx = core.FileStreamContext(slot, framefilter)
        return ctx
        
    def readBlockTablesIf(self):
        for fsgroup in self.fsgroups:
            fsgroup.readBlockTableIf()

    def readBlocksIf(self):
        for fsgroup in self.fsgroups:
            fsgroup.reqBlocks()

    def new_block_cb__(self, valkkafs: ValkkaFS):
        """Called when block information 
        in valkkafs has been updated
        """
        pass
        """TODO
        - increase global min/max times if necessary
        - ..if they change, use a callback
        - continue the callback chain if necessary
        """
        
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
                return

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
            
            self.currentmstime = mstime
            
            if self.timecallback is not None:
                try:
                    self.timecallback(mstime)
                except Exception as e:
                    self.logger.warning("timeCallback__ : your callback failed with '%s'", str(e))
                    
        except Exception as e:
            raise(e)
            # self.logger.warning("timeCallback__ failed with '%s'" % (str(e)))
    

    def timeLimitsCallback__(self, tup):
        """Called from cpp side, see:

        ::

            cachestream.cpp
                FileCacheThread
                    switchCache
                        calls pyfunc2 (this method)

        TODO: analyzer cpp-python-cpp callchain
        """
        try:
            self.logger.debug("timeLimitsCallback__ : %s", str(tup))
            self.logger.debug("timeLimitsCallback__ : %s -> %s", formatMstimestamp(tup[0]), formatMstimestamp(tup[1]))
            if self.timelimitscallback is not None:
                self.timelimitscallback(tup)
        except Exception as e:
            print("timeLimitsCallback__ failed with '%s'" % (str(e)))



# "in-situ" trivial tests
def fsgrouptest():
    # create a dummy ValkkaFS
    valkkafs_1 = ValkkaFS.newFromDirectory(
        dirname = "./vfs1",
        blocksize = 1024*1024,
        n_blocks = 5,
        verbose = True)
    cachethread = core.FileCacheThread("cacher")
    # one would set callbacks to cachethread before starting..
    cachethread.startCall() 
    fsgroup = FSGroup(
        valkkafs_1,
        core.ValkkaFSReaderThread("reader", valkkafs_1.core, cachethread.getFrameFilter()),
        core.ValkkaFSWriterThread("writer", valkkafs_1.core)
    )
    fsgroup.start()
    fsgroup.setSlotId(1, 1001)
    in_filter = fsgroup.getInputFilter()
    # .. could connect lots of streams into in_filter
    out_filter = core.InfoFrameFilter("info")
    file_stream_ctx = fsgroup.setOutputFilter(filter = out_filter, slot = 1)
    cachethread.registerStreamCall(file_stream_ctx)
    fsgroup.clearSlotId(1)
    fsgroup.stop() # writes to cachethread's filter, so must be stopped first
    cachethread.stopCall()


def managertest():
    # create a dummy ValkkaFS
    valkkafs_1 = ValkkaFS.newFromDirectory(
        dirname = "./vfs1",
        blocksize = 1024*1024,
        n_blocks = 5,
        verbose = True)
    valkkafs_2 = ValkkaFS.newFromDirectory(
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
    # valkkafs_1 map internally slot 1 to id 1 on disk
    # output cacherthread goes into out_filter_1:
    file_stream_ctx_1 = manager.map_(
        valkkafs = valkkafs_1,
        framefilter = out_filter_1,
        slot = 1,
        _id = 1)
    file_stream_ctx_2 =manager.map_(
        valkkafs = valkkafs_2,
        framefilter = out_filter_2,
        slot = 2,
        _id = 2)
    # remember that is possible to write 
    # multiple streams into the same ValkkaFS


if __name__ == "__main__":
    # fsgrouptest()
    managertest()

