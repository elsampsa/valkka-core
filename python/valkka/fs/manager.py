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
        # setLogger(self.logger, logging.DEBUG) # TODO: unify the verbosity/logging somehow
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
        self.prevtime = 0
        self.state_interval = 1 # refresh state max every N secs
        self.is_playing = False


    def setLogLevel(self, level):
        getLogger(self.logger, level = level)
        for fsgroup in self.fsgroups:
            fsgroup.setLogLevel(level)


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
            

    def iterateFsInput(self):
        for valkkafs, fsgroup in self.fsgroup_by_valkkafs.items():
            yield valkkafs, fsgroup.getInputFilter()


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
        self.started = True
        for fsgroup in self.fsgroups:
            fsgroup.start()
        self.update() # initiates the new_block_cb__ from all the way down from ValkkaFS
        # self.readBlockTablesIf()
        

    def close(self):
        self.requestClose()
        self.waitClose()

    def requestClose(self):
        if not self.started:
            return
        self.started = False
        for fsgroup in self.fsgroups:
            fsgroup.requestStop()

    def waitClose(self):
        if not self.started:
            return
        for fsgroup in self.fsgroups:
            # print(">")
            fsgroup.waitStop()


    def setTimeCallback(self, func: callable):
        """Continue the callback chain, originating from FileCacheThread::run
        
        prototype:

        ::

            func(
                mstime = mstime,
                valkkafs = None/ValkkaFS instance
            )

        """
        self.timecallback = func
        
        
    def setTimeLimitsCallback(self, func: callable):
        """Continue the callback chain, originating from FileCacheThread::switchcache

        Callback carries the timelimits of currently cached frames (as a tuple)

        prototype:

        ::

            func(
                timerange = tuple timerange of currently cached frames
                valkkafs = None/ValkkaFS instance
            )

        """
        self.timelimitscallback = func
        
        
    def setBlockCallback(self, func: callable):
        """

        func prototype:

        ::

            func(
                timerange = tuple of global timelimits
                valkkafs = None/ValkkaFS instance
            )
        
        """
        self.new_block_cb = func


    def map_(self, valkkafs=None, framefilter=None,
            write_slot=None, read_slot=None, _id=None):
        """Mapping valkkafs (i.e. file) => output framefilter

        :param framefilter: framefilter for receiving saved stream

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
        

    def updateTime__(self):
        mstime = 0
        for fsgroup in self.fsgroups:
            mstime = max(fsgroup.currentmstime, mstime)
        self.currentmstime = mstime
        if self.currentmstime > 0:
            # at least one of the stream has been seeked to avail frames
            for fsgroup in self.fsgroups:
                if fsgroup.currentmstime==0:
                    # so there's a stream that is inactive because frames
                    # have ran out.  Let's see if we can start it
                    if fsgroup.withinRange(self.currentmstime):
                        # yes we can
                        fsgroup.seek(self.currentmstime)
                        if self.is_playing:
                            fsgroup.play()


    def updateTimeRange__(self):
        """update the global timerange from all blocktables
        """
        timerange = None
        for fsgroup in self.fsgroups:
            self.logger.debug("updateTimeRange__ : %s", fsgroup)
            self.logger.debug("updateTimeRange__ : fsgroup.timerange: %s", 
                formatMstimeTuple(fsgroup.timerange))
            if fsgroup.timerange is None or fsgroup.timerange == (0,0): # empty BT
                continue
            if timerange is None:
                timerange = (fsgroup.timerange[0], fsgroup.timerange[1])
                continue
            timerange = (
                min(fsgroup.timerange[0], timerange[0]), 
                min(fsgroup.timerange[1], timerange[1])
            )
            self.timerange_by_valkkafs[fsgroup.valkkafs] = fsgroup.timerange
        self.timerange = timerange
        self.logger.debug("updateTimeRange__ : final timerange: %s", 
            formatMstimeTuple(self.timerange))


    def updateTimeLimits__(self):
        """update limits of currently cached frames
        """
        mintime = 0
        maxtime = 0
        for fsgroup in self.fsgroups:
            current_timerange = fsgroup.getCurrentTimerange()
            self.logger.debug("updateTimeLimits__: fsgroup current_timerange: %s:%s", fsgroup, current_timerange)
            if (current_timerange == (0,0)): # not set'ed
                continue
            if mintime <=0: mintime = current_timerange[0]
            if maxtime <=0: maxtime = current_timerange[1]
            mintime = min(current_timerange[0], mintime)
            maxtime = max(current_timerange[1], maxtime)
            self.timerange_by_valkkafs[fsgroup.valkkafs] = current_timerange
        self.current_timerange = (mintime, maxtime)
        self.logger.debug("updateTimeLimits__: current_timerange: %s", self.current_timerange)


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
        self.logger.debug("new_block_cb__: %s", fsgroup)
        """
        if self.new_block_cb is not None:
            self.new_block_cb(
                timerange = fsgroup.getTimerange(),
                valkkafs = fsgroup.valkkafs
            )
        """
        """instead of just informing the timer, we could
        send individual information about each ValkkaFS to the widget
        for example if we'd like to have a timeline per each camera
        """
        self.timer()


    def timeCallback__(self, fsgroup, mstime: int):
        """This originates from the cacherthread of the fsgroup in question.
        """
        self.logger.debug("timeCallback__: %s", fsgroup)
        self.logger.debug("timeCallback__: %s", mstime)
        """
        self.currentmstime = mstime # can't do this..!
        if self.timecallback is not None:
            self.logger.debug("timeCallback__: subcallback:")
            self.timecallback(
                mstime = mstime, 
                valkkafs = fsgroup.valkkafs
            )
        """
        self.timer()


    def timeLimitsCallback__(self, fsgroup, tup):
        """Currently cached frames

        :param fsgroup: FSGroup where this callback is originating
        :param tup: originates from FileCacherThread => FSGroup

        """
        self.logger.debug("timeLimitsCallback__: %s:%s", fsgroup,tup)
        """
        if self.timelimitscallback is not None:
            self.logger.debug("timeLimitsCallback__: subcallback: %s", fsgroup)
            self.timelimitscallback(
                timerange = fsgroup.getCurrentTimerange
                valkkafs = fsgroup.valkkafs
                )
        """
        self.timer()


    def timer(self):
        """Manager should take care the case where one of the groups should pick up again playing
        that has been stopped due to unavailable stream
        """
        t = time.time()
        if ((t - self.prevtime) >= self.state_interval):
            self.logger.debug("timer: state update")
            self.prevtime = t
            self.updateTimeRange__()
            self.updateTimeLimits__()
            self.updateTime__()
            if self.new_block_cb is not None:
                self.new_block_cb(timerange = self.timerange)
            if self.timecallback is not None:
                self.timecallback(mstime = self.currentmstime)
            if self.timelimitscallback is not None:
                self.timelimitscallback(timerange = self.current_timerange)
                    
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
        self.is_playing=True
        for fsgroup in self.fsgroups:
            fsgroup.play()

    def stop(self):
        """stop all fsgroup(s)
        """
        self.is_playing=False
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

