## Design principles

Hierarchical objects, i.e. objects owning objects owning objects:
```
Obj1
    Obj2
        Obj3
```
- Callbacks emanate from the deeper level objects (i.e. Obj3->Obj2->Obj1)
- At each level, a the callbacks change the state of the object
- Higher level objects do not change state of the deeper level objects, instead, they just use deeper level object's getters
- Each object has a method named "update" that calls method "update" of the deeper level object.  This method can be used to trigger a callback
for the first time (in order to propagate the state update up the tree)

This way we achieve "HIMO" (hierarchical & minimally interacting objects).

The concrete object tree:
```
ValkkaFSManager
    FSGroup
        ValkkaFSReaderThread
        ValkkaFSWriterThread
        FileCacherThread
            - sends callbacks (timeCallback__, timeLimitsCallback__)
        ValkkaMultiFS
            core.ValkkaFS
                - sends a callback (new_block_cb__)
```
Manager has an API that can be used to interact with widgets
 

## ValkkaFSManager

```
- Given a list of valkka.fs.base.ValkkaFS (a) instances => creates FSGroup (c) for each one
- A list of FSGroup (c) instances, one per ValkkaFS (a)
    - Each FSGroup encapsulates a core.ValkkaFSReaderThread, core.ValkkaFSWriterThread
      and core.FileCacherThread instance
    - each core.ValkkaFSReaderThread outputs to FileCacherThread's (++) input filter

    Schematically:

    ::

        core.ValkkaFSWriterThread ---> core.ValkkaFS ---> core.ValkkaFSReaderThread ---> core.FileCacherThread
                             slot-to-id              id-to-slot

- Vars
    - timerange: global timerange (not currently cached, but global)
    - currentmstime: current seek/playtime

- API (call after start)

    update()
        - initiate some callbacks from the deeper object in order to populate state

    getInputFilter(valkkafs)
        - returns input filter for correct ValkkaFSWriterThread as per valkkafs

    getTimeRange -> tuple (global timerange)
        - global (not cached frames) timerange
        - originates from the blocktables of each FSGroup -> ValkkaFS
    
    getTimeRangeByValkkaFS(valkkafs) -> tuple
        - get global timerange as per valkkafs

    setTimeCallback
        - set continuation to timeCallback__.  carries current mstime
    
    setTimeLimitsCallback
        - set continuation to timeLimitsCallback__.  
        - timeLimitsCallback__ originates from core.FileCacherThread
        - carries a tuple of timelimits of currently cached frames
    
    setBlockCallback
        - set continuation to new_block_cb__
        - originates from core.ValkkaFS
        - model: func(valkkafs=, timerange_local=, timerange_global=)

    map_(valkkafs, framefilter, write_slot, read_slot, _id)
        - calls map_ of correct FSGroup
        - FSGroup's map_ sets slot-to-id & id-to-slot mappings
        - framefilter is the output of frames from the cacherthread

    unmap(_id)
        - call unmap of correct FSGroup


- Methods

    updateTimeRange__()
        - check each FSGroup's timerange & collate a global timerange
        - updates state: self.timerange

    readBlockTableIf(fsgroup)
        -> calls fsgroup.readBlockTableIf
        - calculate new timerange
        DO NOT USE

    readBlockTablesIf
        -> calls each FSGroup's (c) readBlockTableIf
        - calculate new timerange
        DO NOT USE

    reqBlocksIf
        -> calls each FSGroup's (c) reqBlocksIf
        - FSGroup.reqBlocksIf should be smart enough to exit immediately
        if same blocks are requested again or if out of bounds
        DO NOT USE

    getCurrentTime
        -returns self.currentmstime
    
- Callbacks (connected at ctor)
    timeCallback__ : "heartbeat" from core.FileCacheThread (b)
        - update state: self.currentmstime

    timeLimitsCallback__ : from core.FileCacheThread (b)
        - triggered when the time range of **currently cached frames** has changed
        - composes limits of cached frames from all FSGroups' limits
        - updates state: self.current_timerange

    new_block_cb__(fsgroup) : a notification from FSGroup that there are new frames / blocks
        - carries the FSGroup (c) instance in question
        - calls self.updateTimeRange__() --> updates state: self.timerange


```
## ValkkaFS (a)
```
- Two classes: ValkkaMultiFS, ValkkaSingleFS
- Encapsulates core.ValkkaFS
- Helpers for instantiating core.ValkkaFS, based on json saved data, etc.

- Vars
    blocktable : a copy of the blocktable from cpp side

- Methods
    getBlockTable : updates self.blocktable
    getTimeRange  : inspect blocktable & returns the global timerange for this ValkkaFS
    getInd        : get block indices corresponding to requested timerange
                    - lot's of logic for getting the correct indices: void blocks, wrapping, etc.
    getIndNeigh   : get neighbourhood block indices
                    - again, lot's of logic

    setBlockCallback : set the continuation of new_block_cb__


- Callbacks (connected at ctor)
    new_block_cb__
        - triggered when a new block has been created
        - variant 1: trigger also when there is "significant" amount of new frames
        - calls self.getBlockTable() --> udpdates self.blocktable

- Testing
    - getTimerange, getInd, getIndNeigh tested with separate python notebooks
    - test for void blocks, wrapping, etc.

```
## core.FileCacheThread (b)
```
```
## FSGroup (c)
```
- Encapsulates a core.ValkkaFSReaderThread, core.ValkkaFSWriterThread and core.FileCacherThread instances

- Vars
    - timerange: global timerange for the ValkkaFS/blocktable
    - current_timerange: currently cached frames

- Methods

    map_(write_slot=, read_slot=, _id=, framefilter=)
        -> sets slot <-> id mapping for reader & writer threads
        - incoming frames with slot write_slot are written to disk with id _id
        - frames with id _id on the disk are mapped to slot read_slot when sent downstream
        - creates a FileStreamContext that can be used with FileCacheThread
          for read_slot -> framefilter mapping

    unmap(_id)
        - unmaps

    readBlockTableIf:
        - just calls readBlockTable

    readBlockTable: 
        - state update: self.timerange (from self.valkkafs.getTimeRange() --> from blocktable)
    
    reqBlocksIf(mstime)
        sends a request for new blocks if needed (as judged by mstime)
        -> calls ValkkaFS.getIndNeigh, core.ValkkaFSReaderThread.pullBlocksPyCall
        => frames are streamed into core.FileCacheThread (b)

- Callbacks (connected at ctor)
    new_block_cb__
        - Continuation of ValkkaFS (a) new_block_cb__
        - calls readBlockTableIf

```

Callback scheme:
```

custom callback carrying mstime
    <--ValkkaFSManager.timeCallback(self, fsgroup, mstime) 
        <--FSGroup.timeCallback__(self, mstime)
            <--core.FileCacherThread: callback with mstime

custom callback carrying timelimit tuple
    <--ValkkaFSManager.timeLimitsCallback(self, fsgroup, tup)
        <--FSGroup.timeLimitsCallback__(self, tup)
            <--core.FileCacherThread: callback with tuple (timelimits of currently cached frames)

custom callback (valkkafs=, timerange_local=, timerange_global=)
    <--ValkkaFSManager.new_block_cb__(self, fsgroup)
        <--ValkkaMultiFS.new_block_cb__(self)
            <--core.ValkkaFS: callback when a new block is created

```


## Diagrams
```
```

## TODO
- minimal test with a single rtsp stream: ValkkaFSManager API minimal test
=> test global timerange callbacks, etc.
- notebooks for ValkkaFS

1. quickfix playback
2. ValkkaSingleFS & test
3. Test "holey" video: 
4. with test_studio_5.py playback-mode only: record, quit, wait, record, go to playback-only mode (start with no_rec)


## EXTRA TODO

Test valkka shmem intercom between containers:

https://stackoverflow.com/questions/56878405/sharing-memory-across-docker-containers-ipc-host-vs-ipc-shareable


## Holey' video

```
blocks: |------|--------------|----------------------|

frames: **********         ********                ***
                  X                X


FileCacherThread should emit a special frame (X) when it observes a long delay between frames
```

