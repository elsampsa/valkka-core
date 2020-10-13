"""base.py : A simple multiprocessing framework with back- and frontend and pipes communicating between them

Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen.

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
@version 1.0.1 

@brief   A simple multiprocessing framework with back- and frontend and pipes communicating between them
"""
from multiprocessing import Process, Pipe
import select
import errno
import time
import sys, signal, os, pickle, math
import logging
import asyncio

from valkka.api2.tools import getLogger, setLogger

logger = getLogger(__name__)


def safe_select(l1, l2, l3, timeout = None):
    """
    print("safe_select: enter")
    select.select(l1,l2,l3,0)
    print("safe_select: return")
    return True
    """
    try:
        if timeout is None:
            res = select.select(l1, l2, l3) # blocks
        else:
            res = select.select(l1, l2, l3, timeout) # timeout 0 is just a poll
        # print("safe_select: ping")
    except (OSError, select.error) as e:
        if e.errno != errno.EINTR:
            raise
        else: # EINTR doesn't matter
            # print("select : giving problems..")
            return [[], [], []]  # dont read socket
    else:
        # print("select : returning true!")
        return res  # read socket
    # print("select : ?")



class MessageObject:

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs

    def __str__(self):
        return "<MessageObject: %s: %s>" % (self.command, self.kwargs)

    def __getitem__(self, key):
        return self.kwargs[key]


class MessageProcess(Process):
    """This Process class has been "ripped" from valkka-live (https://github.com/elsampsa/valkka-live) where we initially
    tested this concept.

    The class encapsulated:

    - Frontend methods (in the current main process).  These write to the intercom pipe self.front_pipe (see below).
    - Backend methods (that run in the background/forked process).  These write to self.back_pipe (see below).
    - Intercom pipes that communicate between front- and backend

    Pipes are named as follows:


    ::

        self.front_pipe      : write messages to multiprocess / read messages from multiprocess
        self.back_pipe       : multiprocess reads / writes messages to frontend

    - Multiprocess expects MessageObject instances from self.back_pipe
    - MessageObject coming from self.back_pipe is mapped into a backend method
    
    - Backend sends MessageObject instances to self.back_pipe => self.front_pipe
    - These are being read by the rest of the program
    """

    timeout = 1.0
    # **** define here your backend methods
    # **** 

    def c__ping(self, lis = []):
        """A demo backend method that corresponds to a MessageObject
        """
        print("c__ping:", lis)
        self.send_out__(MessageObject("pong", lis = [1,2,3]))

    # ****


    def __init__(self, name = "MessageProcess"):
        self.name = name
        self.pre = self.__class__.__name__ + "." + self.name
        self.logger = getLogger(self.pre)
        super().__init__()
        self.front_pipe, self.back_pipe = Pipe() # incoming messages
        self.loop = True
        self.listening = False # are we listening something else than just the intercom pipes?
        self.sigint = True

    def ignoreSIGINT(self):
        self.sigint = False

    def __str__(self):
        return "<"+self.pre+">"

    def setDebug(self):
        setLogger(self.logger, logging.DEBUG)

    def preRun__(self):
        pass

    def postRun__(self):
        pass

    # **** backend ****

    def run(self):
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
        """Multiplex all intercom pipes
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
        """Pickle obj & send to outgoing pipe
        """
        # print("send_out__", obj)
        self.back_pipe.send(obj) # these are mapped to Qt signals



    # **** frontend ****

    def getPipe(self):
        return self.front_pipe

    def sendMessageToBack(self, message: MessageObject):
        self.front_pipe.send(message)

    def go(self):
        self.start()

    def requestStop(self):
        self.sendMessageToBack(None)
        
    def waitStop(self):
        self.join()

    def stop(self):
        self.requestStop()
        self.waitStop()

    # *** your frontend methods ***

    def sendPing(self, lis):
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

    def __init__(self, read_fd, write_fd):
        # file descriptors, i.e. numbers:
        self.read_fd = read_fd
        self.write_fd = write_fd
        # these are _io.FileIO objects:
        self.reader = os.fdopen(read_fd, "br", buffering = 0)
        self.writer = os.fdopen(write_fd, "bw", buffering = 0)
    
    def getReadIO(self):
        """_io.FileIO object
        """
        return self.reader

    def getReadFd(self):
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


class AsyncBackMessageProcess(MessageProcess):
    """Normal frontend, asynchronous backend
    """
    
    def __init__(self, name = "AsyncMessageProcess"):
        self.name = name
        self.pre = self.__class__.__name__ + "." + self.name
        self.logger = getLogger(self.pre)
        super().__init__()
        # self.front_pipe, self.back_pipe = getPipes(True, False) # blocking frontend, non-blocking backend (for asynchronous backend)
        self.front_pipe, self.back_pipe = getPipes(True, True) # both blocking: for testing # seems to make no difference (asyncio sets the pipes to non-blocking mode)
        self.loop = True
        self.listening = False # are we listening something else than just the intercom pipes?
        self.sigint = True


    async def c__ping(self, lis = []):
        """A demo backend method that corresponds to a MessageObject

        So, here do await, launch several parallel asyncio tasks, etc.
        """
        print("c__ping:", lis)
        await self.send_out__(MessageObject("pong", lis = [1,2,3]))


    def run(self):
        if self.sigint == False:
            signal.signal(signal.SIGINT, signal.SIG_IGN) # handle in master process correctly
        self.preRun__()
        asyncio.get_event_loop().run_until_complete(self.async_run__())
        """
        print("reading from", self.back_pipe.read_fd)
        back_reader = self.back_pipe.getReadIO()
        print("getting some")
        b = back_reader.read(4096)
        print("got some", b)
        """
        self.postRun__()


    async def asyncPre__(self):
        pass

    async def asyncPost__(self):
        pass

    async def async_run__(self):
        await self.asyncPre__()

        # print("hello from async")
        loop = asyncio.get_event_loop()

        # arrange reading of the intercom pipe
        back_reader = self.back_pipe.getReadIO()
        back_writer = self.back_pipe.getWriteIO()

        self.stream_reader = asyncio.StreamReader()
        def protocol_factory():
            return asyncio.StreamReaderProtocol(self.stream_reader)
            
        self.reader_transport, pro = await loop.connect_read_pipe(protocol_factory, back_reader)
        self.writer_transport, pro = await loop.connect_write_pipe(asyncio.BaseProtocol, back_writer)

        # print(">>", self.writer_transport)

        while self.loop:
            # watch file descriptors: loop.add_reader(fd, callback, *args)
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
        rlis = [self.back_pipe]
        r, w, e = safe_select(rlis, [], [], timeout = timeout) # timeout = 0 == this is just a poll
        # handle the main intercom pipe
        if self.back_pipe in r:
            self.handleBackPipe__(self.back_pipe)    
            r.remove(self.back_pipe)
        # in your subclass, handle rest of the pipes


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
        # print("routeMainPipe__")
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
        """Pickle obj & send to outgoing pipe
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


    def sendMessageToBack(self, message: MessageObject):
        # print("writing to", self.front_pipe.write_fd)
        self.front_pipe.send(message)



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

