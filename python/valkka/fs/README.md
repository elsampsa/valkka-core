 
## ValkkaFSManager
```
- Given a list of valkka.fs.base.ValkkaFS (a) instances => creates FSGroup (c) for each one
- Encapsulates a core.FileCacheThread (b) instance: caches all (encoded) 
    frames from all streams managed by the ValkkaFSManager
- A list of FSGroup (c) instances, one per ValkkaFS (a)
    - Each FSGroup encapsulates a core.ValkkaFSReaderThread and core.ValkkaFSWriterThread instance
    - ..each core.ValkkaFSReaderThread outputs to FileCacheThread's (++) input filter

- Vars
    - timerange: global timerange (not currently cached, but global)
    - currentmstime: current seek/playtime

- API (call after start)

    getInputFilter(valkkafs)

    getTimeRange -> tuple (global timerange)
    
    getTimeRangeByValkkaFS(valkkafs) -> tuple

    setTimeCallback
        - set continuation to timeCallback__.  carries current mstime
    
    setTimeLimitsCallback
        - set continuation to timeLimitsCallback__.  
          carries a tuple of timelimits of currently cached frames
    
    setBlockCallback
        - set continuation to new_block_cb__
          model: func(valkkafs=, timerange_local=, timerange_global=)

    map_(valkkafs, framefilter, write_slot, read_slot, _id)
        - call map_ of correct FSGroup

    unmap(_id)
        - call unmap of correct FSGroup


- Methods
    readBlockTableIf(fsgroup)
        -> calls fsgroup.readBlockTableIf
        - calculate new timerange

    readBlockTablesIf
        -> calls each FSGroup's (c) readBlockTableIf
        - calculate new timerange

    reqBlocksIf
        -> calls each FSGroup's (c) reqBlocksIf
        - FSGroup.reqBlocksIf should be smart enough to exit immediately
        if same blocks are requested again or if out of bounds

    getCurrentTime: returns self.currentmstime
    
- Callbacks (connected at ctor)
    timeCallback__ : "heartbeat" from core.FileCacheThread (b)
        - sets self.currentmstime
        -> calls readBlockTablesIf, reqBlocksIf

    timeLimitsCallback__ : from core.FileCacheThread (b)
        - Triggered when the time range of **currently cached frames** has changed

    new_block_cb__(fsgroup) : a notification from FSGroup that there are new frames / blocks
        - carries the FSGroup (c) instance in question
        - calls readBlockTableIf(fsgroup.valkkafs)


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

- Testing
    - getTimerange, getInd, getIndNeigh tested with separate python notebooks
    - test for void blocks, wrapping, etc.

```
## core.FileCacheThread (b)
```
```
## FSGroup (c)
```
- Encapsulates a ValkkaFS (a), reader and writer threads

- Vars
    - timerange: global timerange for the ValkkaFS/blocktable in question

- Methods

    map_(write_slot=, read_slot=, _id=, framefilter=)
        -> sets slot <-> id mapping for reader & writer threads
        - incoming frames with slot write_slot are written to disk with id _id
        - frames with id _id on the disk are mapped to slot read_slot when sent downstream
        - creates a FileStreamContext that can be used with FileCacheThread
          for read_slot -> framefilter mapping

    getFileStreamContext(_id)
        - ..get that FileStreamContext

    unmap(_id)
        - unmaps

    readBlockTableIf:
        -> calls readBlockTable
        - avoid calling readBlockTable too frequently
    readBlockTable: 
        -> calls ValkkaFS.getBlockTable, getTimeRange
        - updates BT (in ValkkaFS) and self.timerange
    reqBlocksIf(mstime)
        sends a request for new blocks if needed (as judged by mstime)
        -> calls ValkkaFS.getIndNeigh, core.ValkkaFSReaderThread.pullBlocksPyCall
        => frames are streamed into core.FileCacheThread (b)

- Callbacks (connected at ctor)
    new_block_cb__
        - Continuation of ValkkaFS (a) new_block_cb__

```

## Diagrams
```
ValkkaFSManager

    cacherthread
        ---> slot 1 output to ff A
        ---> slot 2 output to ff B
        ---> slot 3 output to ff C
        ...

        slot N to ff mappings defined by registerStreamCall(FileStreamContext)

    FSGroup1
        valkkafs
        reader ---> cacherthread
        writer <--- from input filter
        - writer: input slot => id
        - reader: id => output slot
        - slot => ff mapping for the cacherthread
        - file_stream_ctx_by_slot = {}

    FSGroup2
        valkkafs
        reader ---> cacherthread
        writer 

    FSGroup3
        valkkafs
        reader ---> cacherthread
        writer 

    FSGroup4
        valkkafs
        reader ---> cacherthread
        writer
```



## TODO
- minimal test with a single rtsp stream: ValkkaFSManager API minimal test
=> test global timerange callbacks, etc.
- test_studio_5 with the new API (see that the API makes sense)
- .. & translate code to ValkkaMultiFS
- Start thinking about ValkkaSingleFS ..

## Testing

Testing scheme for this monster:

- Notebooks (for ValkkaFS (a))
- Stream & see that global timerange is updated correctly
