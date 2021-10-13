import time, logging
from valkka import core # ValkkaFSWriterThread, ValkkaFSReaderThread, FileCacheThread # api level 1
from valkka.fs.group import FSGroup
from valkka.fs.multi import ValkkaMultiFS
from valkka.fs.single import ValkkaSingleFS
from valkka.fs.tools import formatMstimeTuple
from valkka.api2.tools import getLogger, setLogger


class ValkkaFSManager:
    """Manages a group of FSGroup {FSGroup = (reader, writer, cacher)} instances

    :param valkafs_list: A list of api level 2 ValkkaFS objects

    Commands all FSGroup's in unison to produce sync'ed playback
    """
    def __init__(self, valkkafs_list: list, name = "manager"):
        self.logger = getLogger(self.__class__.__name__)
        setLogger(self.logger, logging.DEBUG)
        self.name = name
        self.fsgroups = []
        self.fsgroup_by_valkkafs = {}
        self.timerange_by_valkkafs = {}

        self.new_block_cb = None
        self.timecallback = None
        self.timelimitscallback = None

        """State

        - actively updated (only) by callbacks
        - queried through the API
        """

        """Global timerange
        """
        self.timerange = (0, 0) # queried by, say, a widget

        """Currently cached frames timerange
        """
        self.current_timerange = (0,0)

        for valkkafs in valkkafs_list:
            fsgroup = FSGroup(valkkafs)
            fsgroup.setBlockCallback(self.new_block_cb__)
            fsgroup.setTimeCallback(self.timeCallback__)
            fsgroup.setTimeLimitsCallback(self.timeLimitsCallback__)
            self.fsgroups.append(fsgroup)
            self.fsgroup_by_valkkafs[valkkafs] = fsgroup
            self.timerange_by_valkkafs[valkkafs] = (0,0)
        self.started = False


    def __del__(self):
        self.close()


    def update(self):
        for fsgroup in self.fsgroups:
            fsgroup.update()


    def getInputFilter(self, valkkafs = None):
        if valkkafs is None:
            assert(len(self.fsgroups)==1), "ValkkaFS must be provided"
            fsgroup = self.fsgroups[0]
        else:
            fsgroup = self.fsgroup_by_valkkafs[valkkafs]
        return fsgroup.getInputFilter()
            

    def getTimeRange(self):
        """API end-point for widgets, etc.
        """
        # self.updateTimeRange__() # nopes!
        # getters should _not_ update anything
        return self.timerange


    def getTimeRangeByValkkaFS(self, valkkafs):
        """API end-point for widgets, etc.
        """
        return self.timerange_by_valkkafs[valkkafs]


    def start(self):
        for fsgroup in self.fsgroups:
            fsgroup.start()
        self.update() # initiates the new_block_cb__ from all the way down from ValkkaFS
        # self.readBlockTablesIf()
        

    def close(self):
        if not self.started:
            return
        for fsgroup in self.fsgroups:
            fsgroup.requestStop()
        for fsgroup in self.fsgroups:
            fsgroup.waitStop()
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
        """Mapping valkkafs (i.e. file) => output framefilter

        (the output comes from cacherthread)

        User needs to make up an id.  It can be identical
        to the slot number

        ::

            writerthread --> ValkkaFS --> readerthread --> cacherthread --> framefilter

        """
        fsgroup = self.fsgroup_by_valkkafs[valkkafs]
        fsgroup.map_(
            read_slot = read_slot,
            write_slot = write_slot,
            _id = _id,
            framefilter = framefilter
        )
        

    def unmap(self, valkkafs=None, _id=None):
        fsgroup = self.fsgroup_by_valkkafs[valkkafs]
        fsgroup.unmap(_id)
        

    def updateTimeRange__(self):
        """update the global timerange from all blocktables
        """
        timerange = None
        for fsgroup in self.fsgroups:
            self.logger.debug("updateTimeRange__ : fsgroup.timerange: %s", 
                formatMstimeTuple(fsgroup.timerange))
            if fsgroup.timerange is None: # empty BT
                continue
            if timerange is None:
                timerange = (fsgroup.timerange[0], fsgroup.timerange[1])
                continue
            timerange[0] = min(fsgroup.timerange[0], timerange[0])
            timerange[1] = max(fsgroup.timerange[1], timerange[1])
            self.timerange_by_valkkafs[fsgroup.valkkafs] = fsgroup.timerange
        self.timerange = timerange
        self.logger.debug("updateTimeRange__ 2: fsgroup.timerange: %s", 
            formatMstimeTuple(fsgroup.timerange))


    def readBlockTableIf(self, fsgroup):
        """Re-read a single blocktable, update timerange

        This is done earlier in the callback chain, so don't use this
        in new_block_cb__
        i.e. don't request state change for lower level object,
        just use getters
        """
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
        self.logger.debug("new_block_cb__")
        # self.readBlockTableIf(fsgroup) # nopes .. done earlier in the callback chain
        self.updateTimeRange__()
        if self.new_block_cb is not None:
            self.new_block_cb(
                valkkafs = fsgroup.valkkafs,
                timerange_local = fsgroup.timerange,
                timerange_global = self.timerange
            )


    def timeCallback__(self, fsgroup, mstime: int):
        """This originates from the cacherthread of the fsgroup in question.
        
        We can check if all the streams are approx. in the same time
        Create a mean value of all times..?
        """
        self.logger.debug("timeCallback__: %s", mstime)
        self.currentmstime = mstime
        if self.timecallback is not None:
            self.logger.debug("timeCallback__: subcallback")
            self.timecallback(mstime)


    def timeLimitsCallback__(self, fsgroup, tup):
        """Currently cached frames
        """
        self.logger.debug("timeLimitsCallback__: %s", tup)
        mintime = 0
        maxtime = 0
        for fsgroup in self.fsgroups:
            current_timerange = fsgroup.getCurrentTimerange()
            self.logger.debug("timeLimitsCallback__: fsgroup current_timerange: %s", current_timerange)
            if current_timerange == (0,0): # not setted
                continue
            if mintime <=0: mintime = current_timerange[0]
            if maxtime <=0: maxtime = current_timerange[1]
            mintime = min(current_timerange[0], mintime)
            maxtime = max(current_timerange[1], maxtime)
            self.timerange_by_valkkafs[fsgroup.valkkafs] = current_timerange
        self.current_timerange = (mintime, maxtime)
        self.logger.debug("timeLimitsCallback__: current_timerange: %s", self.current_timerange)
        if self.timelimitscallback is not None:
            self.logger.debug("timeLimitsCallback__: subcallback")
            self.timelimitscallback(self.current_timerange)

    # getters

    def getCurrentTime(self):
        return self.currentmstime
   
    def getTimerange(self):
        return self.timerange

    def getCurrentTimerange(self):
        return self.current_timerange

    # play/stop/seek control

    def play(self):
        """play all fsgroup(s)
        """
        for fsgroup in self.fsgroups:
            fsgroup.play()

    def stop(self):
        """stop all fsgroup(s)
        """
        for fsgroup in self.fsgroups:
            fsgroup.stop()   
    
    def seek(self, mstimestamp):
        """seek all fsgroup(s)
        """
        for fsgroup in self.fsgroups:
            ok = fsgroup.seek(mstimestamp)
        


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
    # remember that it's also possible to write 
    # multiple streams into the same ValkkaFS
    manager.unmap(valkkafs=valkkafs_1, _id=1001)
    manager.unmap(valkkafs=valkkafs_2, _id=2002)


if __name__ == "__main__":
    managertest()

