import time
from valkka import core # ValkkaFSWriterThread, ValkkaFSReaderThread, FileCacheThread # api level 1
# from valkka.fs.base import ValkkaFS # api level 2
from valkka.fs.multi import ValkkaMultiFS
from valkka.fs.single import ValkkaSingleFS


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

    def __init__(self, valkkafs = None, 
        reader: core.ValkkaFSReaderThread = None,
        writer: core.ValkkaFSWriterThread = None):
        assert(valkkafs is not None)
        assert(isinstance(reader, core.ValkkaFSReaderThread))
        assert(isinstance(writer, core.ValkkaFSWriterThread))
        self.valkkafs = valkkafs
        self.reader = reader
        self.writer = writer
        self.started = False
        self.valkkafs.setBlockCallback(self.new_block_cb__)
        self.cb = None # a custom callback
        self.file_stream_ctx_by_slot = {}
        self.current_blocks = [] # currently cached blocks
        self.timerange = None # global timerange from the blocktable
        # .. tuple timerange.  None means empty blocktable
        self.checktime = 0 # when BT was requested for the last time
        self.readBlockTable() # inits timerange & checktime

    def getInputFilter(self):
        return self.writer.getFrameFilter()

    def map_(self, slot = None, _id = None, framefilter = None):
        assert(slot is not None)
        assert(_id is not None)
        assert(framefilter is not None)
        self.writer.setSlotIdCall(slot, _id)
        self.reader.setSlotIdCall(slot, _id)
        self.file_stream_ctx_by_slot[slot] =\
            core.FileStreamContext(slot, framefilter)

    def unmap(self, slot):
        assert(slot in self.file_stream_ctx_by_slot)
        self.writer.unSetSlotIdCall(slot)     
        self.reader.unSetSlotIdCall(slot)
        self.file_stream_ctx_by_slot.pop(slot)
        
    def getFileStreamContext(self, slot):
        assert(slot in self.file_stream_ctx_by_slot)
        return self.file_stream_ctx_by_slot[slot]

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
            self.cb(self) # emit this FSGroup with the callback


    def readBlockTable(self):
        """Create a copy of the blocktable
        """
        self.valkkafs.getBlockTable() # updates BT
        # self.logger.debug("readBlockTable: %s", self.blocktable)
        # self.timerange = self.valkkafs.getTimeRange(self.blocktable)
        self.timerange = self.valkkafs.getTimeRange()
        # print("FSGroup: readBlocktable: timerange=", self.timerange)
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
        if self.timerange is None:
            # blocktable is empty
            return
        # (1) check if mstime falls within the current block timerange
        if (mstime-self.timerange[0]) > self.timetolerance:
            return False
        if (self.timerange[1]-mstime) > self.timetolerance:
            return False
        # .. we're well inside the timelimits
        # otherwise:
        # TODO: check if we're inside the legit time limits
        block_list = self.valkkafs.getIndNeigh(
            n=1, 
            time=mstime)
            # blocktable = self.blocktable)
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


# "in-situ" trivial tests
def fsgrouptest():
    # create a dummy ValkkaFS
    # valkkafsclass = ValkkaMultiFS
    valkkafsclass = ValkkaSingleFS
    valkkafs_1 = valkkafsclass.newFromDirectory(
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
    in_filter = fsgroup.getInputFilter()
    # .. could connect lots of streams into in_filter
    out_filter = core.InfoFrameFilter("info")
    # 
    fsgroup.map_(slot=1, _id=1001, framefilter=out_filter)
    # both writer & reader thread map slot 1 to id 1001
    # creates FileStreamContext mapping slot 1 to out_filter
    # that can be used with FileCacheThread
    # let's get it:
    file_stream_ctx = fsgroup.getFileStreamContext(1)
    # tell cachethread to divert slot 1 to out_filter
    cachethread.registerStreamCall(file_stream_ctx)
    # unmap
    fsgroup.unmap(1)
    # 
    fsgroup.stop() # writes to cachethread's filter, so must be stopped first
    cachethread.stopCall()


if __name__ == "__main__":
    fsgrouptest()

