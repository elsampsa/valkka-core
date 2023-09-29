"""base.py : A simple multiprocessing framework with back- and frontend and pipes communicating between them

Copyright 2017-2023 Sampsa Riikonen

Authors: Sampsa Riikonen (sampsa.riikonen@iki.fi)

This particular file, referred below as "Software", is licensed under the MIT LICENSE:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE 
AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

@file    base.py
@author  Sampsa Riikonen
@date    2020
@version 1.6.1 

@brief   A simple multiprocessing framework with back- and frontend and pipes communicating between them
"""
from multiprocessing import Process, Pipe
import select
import errno
import time
import sys, signal, os, pickle, math
import logging
import asyncio
import traceback

# from valkka.api2.tools import getLogger, setLogger

logger = logging.getLogger(__name__)


def safe_select(l1, l2, l3, timeout = None):
    """Like select.select, but ignores EINTR
    """
    try:
        if timeout is None:
            res = select.select(l1, l2, l3) # blocks
        else:
            res = select.select(l1, l2, l3, timeout) # timeout 0 is just a poll
    except (OSError, select.error) as e:
        if e.errno != errno.EINTR:
            raise
        else: # EINTR doesn't matter
            return [[], [], []]  # dont read socket
    else:
        return res  # read socket


class MessageObject:
    """A generic MessageObject for intercommunication between
    fronend (main python process) and backend (forked multiprocess).
    
    Encapsulates a command and parameters

    :param command: the command
    :param kwargs: kwargs

    Example:

    ::

        msg = MessageObject("a-command", par1=1, par2=2)
        msg() # returns "a-command"
        msg["par1"] # returns 1

    """
    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs

    def __str__(self):
        return "<MessageObject: %s: %s>" % (self.command, self.kwargs)

    def __call__(self):
        return self.command

    def __getitem__(self, key):
        return self.kwargs[key]


class MessageProcess(Process):
    """Encapsulates:

    - Frontend methods (in the current main process)
    - Backend methods (that run in the background/forked process)
    - Intercom pipes that communicate (seamlessly) between the multiprocessing front- and backend
    - All intercom is encapsulated in ``MessageObject`` s

    When you send a ``MessageObject`` with command ``myStuff``, the forked multiprocess
    (aka backend) tries to find and execute the method ``c__myStuff`` in the backend.

    :param name: name of the multiprocess

    NOTE: when subclassing ``__init__``, remember to call therein ``super().__init__()``
    """
    timeout = 1.0

    def __init__(self, name = "MessageProcess"):
        self.name = name
        self.pre = self.__class__.__name__ + "." + self.name
        self.logger = logging.getLogger(self.pre)
        super().__init__()
        self.front_pipe, self.back_pipe = Pipe() # incoming messages & pipe that is read by the main pythn process
        self.front_pipe_internal, self.back_pipe_internal = Pipe() # used internally, for example, to wait results from the backend
        self.loop = True
        self.listening = False # are we listening something else than just the intercom pipes?
        self.sigint = True

    @classmethod
    def formatLogger(cls, level = logging.INFO):
        """A helper to setup logger formatter

        Sets loglevel to the automatically created logger ``self.logger`` 
        (that has the name ``classname.name``)

        :param level: loglevel.  Default: ``logging.INFO``.
        """
        logger = logging.getLogger(cls.__name__)
        if not logger.hasHandlers():
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        if level is not None:
            logger.setLevel(level)


    def __str__(self):
        return "<"+self.pre+">"

    def setDebug(self):
        self.logger.setLevel(logging.DEBUG)

    def preRun__(self):
        """Multiprocessing backend method: subclass if needed
        
        Everything that needs to be done *after the fork* (i.e. in the backend), but *before*
        the multiprocesses' main listening & execution loop starts running.

        For example: import heavy libraries and instantiate deep neural net detectors
        """
        pass

    def postRun__(self):
        """Multiprocessing backend method: subclass if needed
        
        Everything that needs to be done *after the fork* (i.e. in the backend), and right *after*
        the multiprocess has exited it's main listening & execution loop, i.e. just before the multiprocess exits
        and dies.

        For example: clear heavy libraries and instantiate deep neural net detectors
        """
        pass

    # **** backend ****

    def run(self):
        """Multiprocessing backend method: the main listening & execution loop.  Normally you would not subclass this one.
        """
        if self.sigint == False:
            signal.signal(signal.SIGINT, signal.SIG_IGN) # handle in master process correctly
        self.preRun__()
        while self.loop:
            if self.listening:
                self.readPipes__(timeout = 0) # timeout = 0 == just poll
            else:
                self.readPipes__(timeout = self.timeout) # timeout of 1 sec
        # indicate front end qt thread to exit
        self.back_pipe.send(None)
        self.logger.debug("bye!")
        self.postRun__()


    def readPipes__(self, timeout):
        """Multiprocessing backend method: listen simultaneously (i.e. "multiplex") all intercom pipes.

        If you need to listen additionally anything else than the normal intercom pipe, please subclass this one.

        :param timeout: listening i/o timeout in seconds
        """
        rlis = [self.back_pipe]
        r, w, e = safe_select(rlis, [], [], timeout = timeout) # timeout = 0 == this is just a poll
        # handle the main intercom pipe
        if self.back_pipe in r:
            self.handleBackPipe__(self.back_pipe)    
            r.remove(self.back_pipe)
        # in your subclass, handle rest of the pipes


    def handleBackPipe__(self, p):
        """Route message to correct method
        """
        ok = True
        try:
            obj = p.recv()
        except Exception as e:
            self.logger.critical("Reading pipe failed with %s", e)
            ok = False
        if ok: 
            self.routeMainPipe__(obj)


    def routeMainPipe__(self, obj):
        """Object from main pipe:
            
        - object.command
        - object.kwargs
        
        => route to "self.c__command(**kwargs)"
        """
        if obj is None:
            self.loop = False
            return

        method_name = "c__%s" % (obj.command)
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            try:
                method(**obj.kwargs)
            except TypeError:
                self.logger.warning("routeMainPipe : could not call method %s with parameters %s" % (method_name, obj.kwargs))
                raise
        else:
            self.logger.warning("routeMainPipe : no such method %s" %(method_name))


    def send_out__(self, obj):
        """Multiprocessing backend method: send an object from the backend to the frontend. 
        It's recommended to use the ``MessageObject`` class.
        """
        self.back_pipe.send(obj)


    def return_out__(self, obj):
        self.back_pipe_internal.send(obj)


    # *** _your_ backend methods ***

    def c__ping(self, lis = []):
        """A demo multiprocessing backend method: triggered when frontend calls the method ``ping`` and sends a reply to frontend
        """
        print("c__ping:", lis)
        self.send_out__(MessageObject("pong", lis = [1,2,3]))


    # **** frontend ****

    def ignoreSIGINT(self):
        """Multiprocessing frontend method: call before ``start`` (or ``go``), so that the multiprocess ignores all SIGINT signals
        """
        self.sigint = False

    def getPipe(self) -> Pipe:
        """Multiprocessing frontend method: returns the pipe you can use to listen to messages sent by the multiprocessing backend.

        returns a ``multiprocessing.Pipe`` instance
        """
        return self.front_pipe

    def sendMessageToBack(self, message: MessageObject):
        """Multiprocessing frontend method: send a ``MessageObject`` to multiprocessing backend
        """
        self.front_pipe.send(message)

    def returnFromBack(self):
        return self.front_pipe_internal.recv()

    def go(self):
        """Multiprocessing frontend method: a synonym to multiprocessing ``start()``
        """
        self.start()

    def requestStop(self):
        """Multiprocessing frontend method: send a request to the multiprocess (backend) to stop
        """
        self.sendMessageToBack(None)
        
    def waitStop(self):
        """Multiprocessing frontend method: alias to multiprocessing ``join()``
        """
        self.join()

    def stop(self):
        """Multiprocessing frontend method: request backend multiprocess to stop and wait until it has finished
        """
        self.requestStop()
        self.waitStop()

    # *** _your_ frontend methods ***

    def sendPing(self, lis):
        """A demo multiprocessing frontend method: a demo method that sends the following ``MessageObject`` to the multiprocessing backend:

        .. code:: python

            MessageObject(
                "ping",
                lis = lis
            ))

        In the backend this is mapped seamlessly into backend method ``c__ping``
        """
        self.sendMessageToBack(MessageObject(
            "ping",
            lis = lis
            ))


# Mixed sync/async processes


def getPipes(block_A = False, block_B = False):
    """

    Either A or B can be blocking or non-blocking 

    non-blocking pipe-terminal is required for asyncio

    ::

        A           B

        w --------> r
        r <-------- w

        (A_w, A_r) is returned as single duplex
    """
    B_read_fd, A_write_fd = os.pipe()
    A_read_fd, B_write_fd = os.pipe()
    #print("read, write pair", B_read_fd, A_write_fd)
    #print("read, write pair", A_read_fd, B_write_fd)
    # these a file descriptors, i.e. numbers

    if block_A:
        os.set_blocking(A_read_fd, True)
        os.set_blocking(A_write_fd, True)
    else:
        os.set_blocking(A_read_fd, False)
        os.set_blocking(A_write_fd, False)

    if block_B:
        os.set_blocking(B_read_fd, True)
        os.set_blocking(B_write_fd, True)
    else:
        os.set_blocking(B_read_fd, False)
        os.set_blocking(B_write_fd, False)

    return Duplex(A_read_fd, A_write_fd), Duplex(B_read_fd, B_write_fd)


def to8ByteMessage(obj):
    b = pickle.dumps(obj)
    # r, w, e = safe_select([], [self.write_fd], [])
    # print("writing to", self.write_fd)
    val = len(b) + 8 # length of the message, including the first 8 bytes
    lenbytes = val.to_bytes(8, byteorder = "big")
    n_pad = math.ceil(val/8)*8 - val
    pad = bytes(n_pad)
    return lenbytes + b + pad


class Duplex:
    """Creates a duplex, similar to what you get from multiprocessing.Pipe(), but other side of the
    duplex is non-blocking (for asyncio backend)
    """
    def __init__(self, read_fd, write_fd):
        # file descriptors, i.e. numbers:
        self.read_fd = read_fd
        self.write_fd = write_fd
        # these are _io.FileIO objects:
        self.reader = os.fdopen(read_fd, "br", buffering = 0)
        self.writer = os.fdopen(write_fd, "bw", buffering = 0)
    
    def fileno(self):
        return self.read_fd

    def getReadIO(self):
        """_io.FileIO object
        """
        return self.reader

    def getReadFd(self):
        """Returns the file descriptor (int), aka "fd" for this pipe
        """
        return self.read_fd

    def getWriteFd(self):
        return self.write_fd

    def getWriteIO(self):
        """_io.FileIO object
        """
        return self.writer

    def recv(self):
        """Traditional blocking recv
        """
        msg = b''
        N = None
        cc = 0
        while True:
            # print("waiting stream")
            res = self.reader.read(8)
            cc += 8
            if N is None:
                # decode the first 8 bytes into int
                N = int.from_bytes(res, byteorder = "big")
                # print("N>", N)
            # print(res, len(res))
            msg += res
            if cc >= N:
                break
        msg = msg[8:N] # remove any padding bytes
        obj = pickle.loads(msg)
        return obj

    def send(self, obj):
        """Tradition blocking send
        """
        msg = to8ByteMessage(obj)
        n = self.writer.write(msg)
        # self.writer.flush() # no effect
        # print("wrote", n, "bytes")
        return n

    def __del__(self):
        self.reader.close()
        self.writer.close()


def exclog(f):
    """Decorator for coroutines: turns exceptions into logging events
    """
    async def wrapper(*args, **kwargs):
        self_ = args[0]
        try:
            return await f(*args, **kwargs)
        except asyncio.CancelledError as e:  # propagate task cancel
            raise (e)
        except Exception as e:  # any other exception should be reported, and BaseException raised so that the program stops
            # raise(BaseException) # enable this if you wan't exceptions raised.  Good for first-stage debugging # DEBUGGING
            self_.logger.critical("asyncio call failed with '%s'", e)
            #self_.logger.critical("------------------------------>")
            self_.logger.critical(traceback.format_exc())
            #self_.logger.critical("<------------------------------")
    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper


def exclogmsg(f):
    """Decorator for coroutines: turns exceptions into logging events
    Sends also an error message to outgoing pipe
    """
    async def wrapper(*args, **kwargs):
        self_ = args[0]
        slot = kwargs["slot"]
        try:
            return await f(*args, **kwargs)
        except asyncio.CancelledError as e:  # propagate task cancel
            raise (e)
        except Exception as e:  # any other exception should be reported, and BaseException raised so that the program stops
            # raise(BaseException) # enable this if you wan't exceptions raised.  Good for first-stage debugging # DEBUGGING
            self_.logger.critical("asyncio call failed with '%s'", e)
            await self_.send_out__(MessageObject("error", slot=slot, error=str(e)))
            #self_.logger.critical("------------------------------>")
            self_.logger.critical(traceback.format_exc())
            #self_.logger.critical("<------------------------------")
    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper



class AsyncBackMessageProcess(MessageProcess):
    """A subclass of ``MessageProcess``, but now the backend runs asyncio

    :param name: multiprocess name

    NOTE: when subclassing ``__init__``, remember to call therein ``super().__init__()``
    """
    def __init__(self, name = "AsyncMessageProcess"):
        self.name = name
        self.pre = self.__class__.__name__ + "." + self.name
        self.logger = logging.getLogger(self.pre)
        super().__init__()
        # self.front_pipe, self.back_pipe = getPipes(True, False) # blocking frontend, non-blocking backend (for asynchronous backend)
        self.front_pipe, self.back_pipe = getPipes(True, True) # both blocking: for testing # seems to make no difference (asyncio sets the pipes to non-blocking mode)
        self.loop = True
        self.listening = False # are we listening something else than just the intercom pipes?
        self.sigint = True

    def getPipe(self) -> Duplex:
        """Returns a Duplex object, instead of multiprocessing.Pipe object.

        Duplex.fileno() returns the read file dtor number
        """
        return self.front_pipe

    def getReadFd(self):
        """Returns read file dtor number for the frontend Duplex object
        """
        return self.front_pipe.getReadFd()

    def getWriteFd(self):
        """Returns write file dtor number for the frontend Duplex object
        """
        return self.front_pipe.getWriteFd()

    def run(self):
        if self.sigint == False:
            signal.signal(signal.SIGINT, signal.SIG_IGN) # handle in master process correctly
        self.preRun__()
        # very important! create a new separate event loop in the forked multiprocess
        loop_ = asyncio.new_event_loop()
        asyncio.set_event_loop(loop_)
        asyncio.get_event_loop().run_until_complete(self.async_run__())
        self.postRun__()


    async def asyncPre__(self):
        """Multiprocessing backend coroutine: subclass if needed
        
        Everything that needs to be done *after the fork* (i.e. in the backend), but *before*
        the multiprocesses' main asyncio event loop starts running.

        In addition to this, you can still subclass also ``preRun__`` that is executed
        after the fork but *before* the asyncio event loop
        """
        pass


    async def asyncPost__(self):
        """Multiprocessing backend coroutine: subclass if needed

        Everything that needs to be done *after the fork* (i.e. in the backend), immediately 
        before exiting the main asyncio event loop

        In addition to this, you can still subclass also ``postRun__`` that is executed
        after exiting the main syncio event loop
        """
        pass


    async def async_run__(self):
        # print("hello from async")
        loop = asyncio.get_event_loop()

        # arrange reading of the intercom pipe
        back_reader = self.back_pipe.getReadIO()
        back_writer = self.back_pipe.getWriteIO()

        self.stream_reader = asyncio.StreamReader()
        def protocol_factory():
            return asyncio.StreamReaderProtocol(self.stream_reader)
            
        """the logic here:

        read input (back_reader) is connected to the event loop, using
        a certain protocol .. protocol == what happens when there is
        stuff to read.  We'll do the reading "manually" in the loop,
        so nothing much

        same for writer.. we get writer_transport where we can write
        """
        self.reader_transport, pro =\
            await loop.connect_read_pipe(protocol_factory, back_reader)
        self.writer_transport, pro =\
            await loop.connect_write_pipe(asyncio.BaseProtocol, back_writer)

        try:
            await self.asyncPre__()
        except Exception as e:
            self.logger.critical(
                "asyncPre__ failed with '%s':\
                don't call anything that requires intercom with the main loop",
                e)

        # ..cause the loop starts over here:
        while self.loop:
            # TODO: we should not have any event/reading loops
            # when using asyncio, so this is a bit stupid
            # solution: define a proper protocol instead
            #
            # if you have file descriptors, add them to the event loop
            # like this: loop.add_reader(fd, callback, *args)
            msg = b''
            N = None
            cc = 0
            while True:
                # print("waiting stream")
                res = await self.stream_reader.read(8)
                cc += 8
                if N is None:
                    # decode the first 8 bytes into int
                    N = int.from_bytes(res, byteorder = "big")
                    # print("N>", N)
                # print(res, len(res))
                msg += res
                if cc >= N:
                    break
            msg = msg[8:N] # remove any padding bytes
            #"""
            #msg = await stream_reader.read(4096)
            obj = pickle.loads(msg)
            # print("obj", obj)
            await self.routeMainPipe__(obj)

        self.reader_transport.close()
        self.writer_transport.close()

        await self.asyncPost__()
        self.logger.debug("bye!")
        

    async def readPipes__(self, timeout):
        """Multiplex all intercom pipes
        """
        raise(BaseException("not used"))


    async def handleBackPipe__(self, p):
        """Route message to correct method
        """
        ok = True
        try:
            obj = p.recv()
        except Exception as e:
            self.logger.critical("Reading pipe failed with %s", e)
            ok = False
        if ok: 
            self.routeMainPipe__(obj)


    async def routeMainPipe__(self, obj):
        """Object from main pipe:
            
        - object.command
        - object.kwargs
        
        => route to "self.c__command(**kwargs)"
        """
        # print("routeMainPipe__", obj)
        if obj is None:
            self.loop = False
            return

        method_name = "c__%s" % (obj.command)
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            # print("method = ", method)
            try:
                await method(**obj.kwargs)
            except TypeError as e:
                self.logger.warning("routeMainPipe : could not call method %s with parameters %s: %s" % (method_name, obj.kwargs, e))
                raise
        else:
            self.logger.warning("routeMainPipe : no such method %s" %(method_name))


    async def send_out__(self, obj):
        """Multiprocessing backend coroutine: pickle obj & send to main python process.
        It's recommended to use the ``MessageObject`` class.
        """
        #print("send_out__", obj, self.writer_transport)
        msg = to8ByteMessage(obj)
        # self.back_pipe.send(obj)
        #try:
        # print("send_out__", msg)
        self.writer_transport.write(msg) # woops.. this is _not_ async call (it returns immediately)
        #except Exception as e:
        #    print("send_out__ failed with", e)
        #print("send_out__: exit")


    # ***  _your_ backend methods ***

    async def c__ping(self, lis = []):
        """A demo backend coroutine:  triggered when frontend calls the method ``ping`` and sends a reply to frontend

        So, in this coroutine it's all asyncio, i.e. await'ing and sending tasks.
        """
        print("c__ping:", lis)
        await self.send_out__(MessageObject("pong", lis = [1,2,3]))


    def sendMessageToBack(self, message: MessageObject):
        # print("writing to", self.front_pipe.write_fd)
        self.front_pipe.send(message)


class MainContext:
    """A convenience class to organize your python main process in the context of multiprocessing

    You should subclass this.  In subclassed ``__init__``, you should always call the
    superclass constructor:

    ::

        def __init__(self):
            # do custom initializations
            # call superclass ctor in the last line of your
            # custom init
            super().__init__()

    This will have the effect of calling ``startProcesses`` and ``startThreads``
    (see below).

    Remember to call the superclass constructor always in the last line of your customized ``__init__``

    Please see tutorial, part II for practical subclassing examples

    ``MainContext`` has a logger ``self.logger`` with the name
    ``classname``.

    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.started = False
        self.aux_pipe_read, self.aux_pipe_write = Pipe()  # when using runAsThread
        self.startProcesses()
        self.startThreads()

    @classmethod
    def formatLogger(cls, level = logging.INFO):
        """A helper to setup logger formatter

        Sets loglevel to the automatically created logger ``self.logger`` 
        (that has the name ``classname``)

        :param level: loglevel.  Default: ``logging.INFO``.
        """
        logger = logging.getLogger(cls.__name__)
        if not logger.hasHandlers():
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        if level is not None:
            logger.setLevel(level)

    def setDebug(self):
        self.logger.setLevel(logging.DEBUG)

    def startProcesses(self):
        """Mandatory. Create, cache and start your ``MessageProcess`` es here.
        """
        raise NotImplementedError("virtual method")

    def startThreads(self):
        """Mandatory. Create, cache and start any python multithreads here if you have them.
        """
        raise NotImplementedError("virtual method")

    def close(self):
        """Mandatory. Terminate all multiprocesses and threads.  Should be
        called in the ``__call__`` method after exiting the main loop.
        """
        raise NotImplementedError("virtual method")

    def __call__(self):
        """Mandatory.  Your main process loop.
        """
        self.loop = True
        while self.loop:
            try:
                time.sleep(1.0)
                print("alive")
            except KeyboardInterrupt:
                print("you pressed CTRL-C: will exit asap")
                break

    def runAsThread(self):
        """Run the class as a thread.  Only for testing/debugging purposes
        """
        from threading import Thread, Event
        self.thread = Thread(target=self.__call__)
        self.logger.critical("starting as thread")
        self.thread.start()  # goes into background

    def stopThread(self):
        """If launched with ``runAsThread``, use this method to stop.
        """
        self.logger.critical("requesting thread stop")
        self.aux_pipe_write.send(None)
        self.thread.join()
        self.close()
        self.logger.critical("thread stopped")



def test1():
    p = MessageProcess()
    p.go()
    pipe = p.getPipe()
    time.sleep(2)
    print("sending ping")
    p.sendPing([1,2,3])
    obj = pipe.recv()
    print("got", obj)
    p.stop()


def test2():
    p = AsyncBackMessageProcess()
    p.go()
    pipe = p.getPipe()
    time.sleep(2)
    # print("sending some to", pipe.write_fd)
    # pipe.send("kokkelis")
    print("sending ping")
    p.sendPing([1,2,3])
    msg = pipe.recv()
    print("=> got", msg)
    print("sending another ping")
    p.sendPing([1,2,3])
    #p.sendPing([1,2,3])
    time.sleep(3)
    print("sending stop")
    p.stop()


if __name__ == "__main__":
    # test1()
    test2()

