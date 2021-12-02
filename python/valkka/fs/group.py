import time, logging, traceback
from valkka import core # ValkkaFSWriterThread, ValkkaFSReaderThread, FileCacheThread # api level 1
# from valkka.fs.base import ValkkaFS # api level 2
from valkka.fs.multi import ValkkaMultiFS
from valkka.fs.single import ValkkaSingleFS
from valkka.fs.tools import formatMstimestamp, formatMstimeTuple
from valkka.api2.tools import getLogger, setLogger


class SlotMapping:
    """Slot mapping for certain frame id in ValkkaFS

    write_slot : incoming slot number that's converted into an id when writing to ValkkaFS
    read_slot  : outgoing slot number that's converted from an id when reading from ValkkaFS
    file_stream_ctx : used with FileCacheThread to do slot => output framefilter mapping
    """
    def __init__(self, write_slot = None, read_slot = None, file_stream_ctx = None):
        self.write_slot = write_slot
        self.read_slot = read_slot
        self.file_stream_ctx = file_stream_ctx


class FSGroup:
    """
    :param valkkafs: ValkkaFS api level 2 instance

    Each FSGroup has a core.FileCacheThread, core.ValkkaFSReaderThread, core.ValkkaFSWriterThread

    The reader dumps into the cacher.  Play, stop, etc. is requested from the cacher.

    Several streams (with unique slot numbers) can be diverted into
    the ValkkaFSWriterThread input filter (they all go into the same ValkkaFS / file).

    When reading, all frames come from the same ValkkaFSReaderThread's 
    (from the same ValkkaFS / file ) output filter.

    Slots are mapped to unique id numbers in the ValkkaFS file (as defined by the user)
    """
    timetolerance = 2000 # if frames are missing at this distance or further, 
    # request for more blocks
    timediff = 5 # blocktable can be inquired max this frequency (secs)

    def __init__(self, valkkafs = None):
        assert(valkkafs is not None)

        logname = self.__class__.__name__
        name = valkkafs.getName()
        if name != "":
            logname = logname + " " + name
        self.logger = getLogger(logname)
        setLogger(self.logger, logging.DEBUG)

        self.valkkafs = valkkafs

        self.cacher = core.FileCacheThread("cacher_" + name)
        self.reader = core.ValkkaFSReaderThread("reader_" + name, self.valkkafs.core, self.cacher.getFrameFilter())
        self.writer = core.ValkkaFSWriterThread("writer_" + name, self.valkkafs.core)

        self.started = False
        self.playing = False

        self.valkkafs.setBlockCallback(self.new_block_cb__)
        self.cacher.setPyCallback(self.timeCallback__)
        self.cacher.setPyCallback2(self.timeLimitsCallback__)

        self.new_block_cb = None # a custom callback
        self.time_cb = None
        self.time_limits_cb = None

        """State

        - is actively updated through callbacks
        """

        """Current set time as reported by FileCacherThread
        0 is the default which means not set

        state req: methods play & getCurrentTime needs
        """
        self.currentmstime = 0

        self.slot_mapping_by_id = {}

        """Currently cached blocks

        state req: if same blocks are requested again, no need
        to request them from the filesystem
        """
        self.current_blocks = []
        
        """global timerange from the blocktable. 
        
        None = empty blocktable:
        """
        self.timerange = None
        
        """currently cached frames.  
        
        (0,0) = no cached frames
        """
        self.current_timerange = (0,0)

        self.checktime = 0 # when BT was requested for the last time
        # self.readBlockTable() # inits timerange & checktime # update initiates this with callbacks

    def __str__(self):
        return "<FSGroup "+self.valkkafs.getName()+">"

    def update(self):
        self.valkkafs.update()

    def getInputFilter(self):
        return self.writer.getFrameFilter()

    def map_(self, write_slot = None, read_slot = None, _id = None, framefilter = None):
        assert(write_slot is not None)
        assert(read_slot is not None)
        assert(_id is not None)
        assert(framefilter is not None)
        self.writer.setSlotIdCall(write_slot, _id)
        self.reader.setSlotIdCall(read_slot, _id)
        file_stream_ctx = core.FileStreamContext(read_slot, framefilter)
        self.slot_mapping_by_id[_id] = SlotMapping(
            write_slot, read_slot, file_stream_ctx
        )
        self.cacher.registerStreamCall(file_stream_ctx)

    def unmap(self, _id):
        # assert(slot in self.file_stream_ctx_by_slot)
        assert(_id in self.slot_mapping_by_id)
        sm = self.slot_mapping_by_id[_id]
        self.writer.unSetSlotIdCall(sm.write_slot)
        self.reader.unSetSlotIdCall(sm.read_slot)
        self.cacher.deregisterStreamCall(sm.file_stream_ctx)
        self.slot_mapping_by_id.pop(_id)
    
    def start(self):
        self.cacher.startCall()
        self.reader.startCall()
        self.writer.startCall()
        self.started = True
        
    def stop(self):
        self.reader.stopCall()
        self.cacher.stopCall()
        self.writer.stopCall()
        self.started = False

    def requestStop(self):
        self.reader.requestStopCall()
        self.cacher.requestStopCall()
        self.writer.requestStopCall()
        self.started = False

    def waitStop(self):
        self.reader.waitStopCall()
        self.cacher.waitStopCall()
        self.writer.waitStopCall()

    # getters

    def getCurrentTime(self):
        return self.currentmstime

    def getTimerange(self):
        return self.timerange

    def getCurrentTimerange(self):
        return self.current_timerange

    # setters

    def clearTime(self):
        """Reset cacher state
        """
        self.logger.debug("clearTime")
        self.cacher.clearCall()

    # play/stop/seek control

    def play(self):
        if self.currentmstime <= 0:
            return False
        self.logger.debug("play")
        self.cacher.playStreamsCall()
        self.playing = True
        return True


    def stop(self):
        if self.playing:
            # traceback.print_stack()
            self.logger.debug("stop")
            self.cacher.stopStreamsCall()
            self.playing = False
        
        
    def seek(self, mstimestamp):
        """Does not request frames if there are already frames in this timerange

        This call originates typically from a GUI widget
        """
        # self.readBlockTableIf() # nopes: state update only by callbacks
        self.logger.debug("seek : target: %s", mstimestamp)
        self.logger.debug("seek : current timerange: %s", self.current_timerange)
        self.logger.debug("seek : current timerange: %s", formatMstimeTuple(self.current_timerange))
        self.logger.debug("seek : current global timerange: %s", self.timerange)
        self.logger.debug("seek : current global timerange: %s", formatMstimeTuple(self.timerange))
        self.logger.debug("seek : timetolerance: %s", self.timetolerance)
        if not self.withinRange(mstimestamp):
            self.logger.debug("seek : not within global range")
            self.clearTime() # this particular stream will be re'setted
            return False

        # accept the current time .. now it's propagaged by self.timer()
        self.currentmstime = mstimestamp

        if self.withinCached(mstimestamp, self.timetolerance):
            self.logger.debug("seek : just set target time")
            self.cacher.seekStreamsCall(mstimestamp, False) 
            # simply sets the target (note that clear is set to False)
            return True
        self.logger.debug("seek : request frames")
        # self.cacher.seekStreamsCall(mstimestamp, True) 
        ## note that clear flag is True : clear the seek.
        ## nopes, we want to go to the frames directly if they are avail
        self.cacher.seekStreamsCall(mstimestamp, False)
        ## similaryly, blocks will be requested only if they are
        ## different from the currently cached ones
        ok = self.reqBlocksIf(mstimestamp)
        if not ok: self.logger.warning("seek : could not get blocks")
        return ok


    # callback setters

    def setBlockCallback(self, cb):
        """Set a callback that is fired when there's a new block/
        the block information is updated

        The callback should accept a single parameter
        """
        self.new_block_cb = cb

    def setTimeCallback(self, cb):
        """Set a callback for cacherthread's "heartbeat"

        The callback should accept two parameter (fsgroup, mstime)
        """
        self.time_cb = cb

    def setTimeLimitsCallback(self, cb):
        """Set callback when cacherthread is filled with new block of frames

        The callback should accept two parameter (fsgroup, tup)
        """
        self.time_limits_cb = cb

    # callbacks

    def new_block_cb__(self):
        """This is always called at a new block
        """
        self.logger.debug("new_block_cb__: ")
        self.readBlockTableIf()
        if self.new_block_cb is not None:
            self.logger.debug("new_block_cb__ : subcallback")
            self.new_block_cb(self) # emit this FSGroup with the callback

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
        # return
        try:
            """
            Handle under / overflows:
            
            - If the stream goes over global timelimits, stop it
            - .. same for underflow
            - If goes under / over currently available blocks, order more
            
            - using try / except blocks we can see the error message even when this is called from cpp
            """
            self.logger.debug("timeCallback__ : ")
            if mstime <= 0: # time not set
                self.logger.debug("timeCallback__ : no time set")
                # print("timeCallback__ : no time set")
            elif self.timerange is None:
                # mstime set but there's no timerange
                self.logger.debug("timeCallback__ : no time range")
                self.clearTime() # a new callback from cacherthread will follow asap
                return
            elif (mstime < self.timerange[0]) or (mstime > self.timerange[1]):
                self.logger.debug("timeCallback__ : out of timerange")
                self.clearTime() # a new callback from cacherthread will follow asap
                return
            elif mstime == self.currentmstime:
                self.logger.debug("timeCallback__ : same time")
            else:
                # request blocks around certain 
                # millisecond timestamp
                self.reqBlocksIf(mstime)
                
            self.currentmstime = mstime
            if self.time_cb is not None:
                try:
                    self.time_cb(self, mstime) # attach fsgroup
                except Exception as e:
                    self.logger.warning("timeCallback__ : your callback failed with '%s'", str(e))
                    traceback.print_exc()

        except Exception as e:
            self.logger.debug("timeCallback__ failed with '%s'" , e)
            traceback.print_exc()


    def timeLimitsCallback__(self, tup):
        try:
            self.logger.debug("timeLimitsCallback__ : %s", str(tup))
            self.logger.debug("timeLimitsCallback__ : %s", formatMstimeTuple(tup))
            if tup is None:
                # FileCacherThread is telling us that there are no cached frames
                # in FSGroup this is indicated with:
                self.current_timerange=(0,0)
                self.current_blocks = []
            else:
                self.current_timerange = tup # currently cached frames
            if self.time_limits_cb is not None:
                self.logger.debug("timeLimitsCallback__ : subcallback")
                # self.time_limits_cb(self, tup) # attach fsgroup
                self.time_limits_cb(self, self.current_timerange)
        except Exception as e:
            self.logger.debug("timeLimitsCallback__ failed with '%s'" , e)
            traceback.print_exc()


    # BT handling

    def readBlockTable(self):
        """Create a copy of the blocktable
        """
        # self.valkkafs.getBlockTable() # updates BT
        # .. nopes: object lower in the hierarchy updates its state
        # .. and higher level objects just use getters of the lower
        # .. level object
        #
        # self.logger.debug("readBlockTable: %s", self.blocktable)
        # self.timerange = self.valkkafs.getTimeRange(self.blocktable)
        self.timerange = self.valkkafs.getTimeRange()
        self.logger.debug("readBlockTable: timerange=%s", self.timerange)
        self.logger.debug("readBlockTable: timerange=%s", formatMstimeTuple(self.timerange))
        self.checktime = time.time()

    def readBlockTableIf(self):
        """Create a copy of the blocktable, but only if a certain time has passed
        """
        self.logger.debug("readBlockTableIf:")
        #if (time.time() - self.checktime) >= self.timediff:
        self.logger.debug("readBlockTableIf: updating")
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

        self.logger.debug("reqBlocksIf: current blocks: %s", 
            self.current_blocks)

        if self.timerange is None:
            # blocktable is empty
            self.logger.debug("reqBlocksIf: empty blocktable")
            return False

        self.logger.debug("reqBlocksIf: mstime: %s, timerange: %s = %s", mstime, self.timerange,
            formatMstimeTuple(self.timerange))

        if not self.withinRange(mstime):
            self.logger.debug("reqBlocksIf: not in range")
            return False

        if self.withinCached(mstime, self.timetolerance):
            self.logger.debug("reqBlocksIf: already within cached frames")
            return False

        block_list = self.valkkafs.getIndNeigh(
            n=1, 
            time=mstime)
            # blocktable = self.blocktable)
        # block numbers are in time order
        # (2) if blocks would be different than the ones we have currently,
        # request them
        if len(block_list) < 1:
            self.logger.debug("reqBlocksIf: got empty list")
            return False
        if block_list == self.current_blocks:
            self.logger.debug("reqBlocksIf: same blocks")
            return False
        # blocks ok
        self.current_blocks = block_list
        tmp = self.valkkafs.blocktable[self.current_blocks,:]
        # NOTE: should we actually wait until FileCacheThread confirms 
        # that self.current_timerange == cache frames..?
        # instead of setting self.current_timerange here
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

    # helpers

    def withinRange(self, mstime):
        """mstimestamp within the range of all avail. frames?
        """
        if ((self.timerange == (0,0)) or self.timerange is None):
            return False
        if (mstime >= (self.timerange[0])) and\
            (mstime <= (self.timerange[1])):
            return True
        return False

    def withinCached(self, mstime, margin):
        """mstimestamp within the range of cached frames?
        """
        if ((self.current_timerange == (0,0)) or self.current_timerange is None):
            return False
        if (mstime >= (self.current_timerange[0] + margin)) and\
            (mstime <= (self.current_timerange[1] - margin)):
            return True
        return False


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
    fsgroup = FSGroup(valkkafs_1)
    fsgroup.start()
    in_filter = fsgroup.getInputFilter()
    # .. could connect lots of streams into in_filter
    out_filter = core.InfoFrameFilter("info")
    # 
    fsgroup.map_(write_slot=1, read_slot=1, _id=1001, framefilter=out_filter)
    # both writer & reader thread map slot 1 to id 1001
    # creates FileStreamContext mapping slot 1 to out_filter
    # that is used internally with FileCacheThread
    fsgroup.unmap(1001)
    fsgroup.stop() # writes to cachethread's filter, so must be stopped first

if __name__ == "__main__":
    fsgrouptest()

