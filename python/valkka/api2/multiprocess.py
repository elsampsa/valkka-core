"""
multiprocess.py : Multiprocessing with a pipe signal scheme

 * Copyright 2017-2020 Valkka Security Ltd. and Sampsa Riikonen
 *
 * Authors: Sampsa Riikonen <sampsa.riikonen@iki.fi>
 *
 * This file is part of the Valkka library.
 * 
 * Valkka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>
 *
 */

@file    multiprocess.py
@author  Sampsa Riikonen
@date    2017
@version 1.3.5 

@brief   Multiprocessing with a pipe signal scheme
"""

# raise(BaseException("use valkka.multiprocess.* instead"))
print("\nWARNING: 'valkka.api2.multiprocess' is DEPRECATED.  Use 'valkka.multiprocess' instead\n")

from multiprocessing import Process, Pipe, Event
import select
import time
from valkka import core
from valkka.api2.tools import *
from valkka.api2.shmem import ShmemRGBClient

pre_mod = "valkka.api2.multiprocess: "


def safe_select(l1, l2, l3, timeout=0):
    """
    print("safe_select: enter")
    select.select(l1,l2,l3,0)
    print("safe_select: return")
    return True
    """
    try:
        if (timeout > 0):
            res = select.select(l1, l2, l3, timeout)
        else:
            res = select.select(l1, l2, l3)
        # print("safe_select: ping")
    except (select.error, v):
        if (v[0] != errno.EINTR):
            print("select : ERROR READING SOCKET!")
            raise
            return [[], [], []]  # dont read socket ..
        else:
            # print("select : giving problems..")
            return [[], [], []]  # dont read socket
    else:
        # print("select : returning true!")
        return res  # read socket
    # print("select : ?")


class ValkkaProcess(Process):
    """
    Semantics:

    Frontend: the part of the forked process that keeps running in the current, user virtual memory space
    Backend : the part of the forked process that runs in its own virtual memory space (e.g. "in the background")

    This class has both backend and frontend methods:

    - Backend methods should only be called from backend.  They are designated with "_".
    - Frontend methods should only be called from frontend

    To avoid confusion, backend methods are designated with "_", except for the "run()" method, that's always in the backend

    Frontend methods use a pipe to send a signal to backend that then handles the signal with a backend method having the same name (but with "_" in the end)

    Backend methods can, in a similar fashion, send signals to the frontend using a pipe.  In frontend, a listening thread is needed.   That thread can then call
    the handleSignal method that chooses the correct frontend method to call

    TODO: add the possibility to bind the process to a certain processor
    """

    # incoming signals : from frontend to backend
    incoming_signal_defs = {  # each key corresponds to a front- and backend methods
        "test_": {"test_int": int, "test_str": str},
        "stop_": []
    }

    # outgoing signals : from back to frontend.  Don't use same names as for
    # incoming signals ..
    outgoing_signal_defs = {
        "test_o": {"test_int": int, "test_str": str},
    }

    def __init__(self, name, affinity=-1, **kwargs):
        super().__init__()
        self.pre = self.__class__.__name__ + " : " + name + \
            " : "  # auxiliary string for debugging output
        self.name = name
        self.affinity = affinity
        self.signal_in = Event()
        self.signal_out = Event()
        # communications pipe.  Frontend uses self.pipe, backend self.childpipe
        self.pipe, self.childpipe = Pipe()

        self.signal_in.clear()
        self.signal_out.clear()

        # print(self.pre, "init")

    def getPipe(self):
        """Returns communication pipe for front-end
        """
        return self.pipe

    def preRun_(self):
        """After the fork, but before starting the process loop
        """
        if (self.affinity > -1):
            os.system("taskset -p -c %d %d" % (self.affinity, os.getpid()))

    def postRun_(self):
        """Just before process exit
        """
        print(self.pre, "post: bye!")

    def cycle_(self):
        # Do whatever your process should be doing, remember timeout every now
        # and then
        time.sleep(5)
        print(self.pre, "hello!")

    def startAsThread(self):
        from threading import Thread
        t = Thread(target=self.run)
        t.start()

    def run(self):  # No "_" in the name, but nevertheless, running in the backed
        """After the fork. Now the process starts running
        """
        # print(self.pre," ==> run")

        self.preRun_()
        self.running = True

        while(self.running):
            self.cycle_()
            self.handleSignal_()

        self.postRun_()

    def handleSignal_(self):
        """Signals handling in the backend
        """
        if (self.signal_in.is_set()):
            signal_dic = self.childpipe.recv()
            method_name = signal_dic.pop("name")
            method = getattr(self, method_name)
            method(**signal_dic)
            self.signal_in.clear()
            self.signal_out.set()

    def sendSignal(self, **kwargs):  # sendSignal(name="test",test_int=1,test_str="kokkelis")
        """Incoming signals: this is used by frontend methods to send signals to the backend
        """
        try:
            name = kwargs.pop("name")
        except KeyError:
            raise(AttributeError("Signal name missing"))

        # a dictionary: {"parameter_name" : parameter_type}
        model = self.incoming_signal_defs[name]

        for key in kwargs:
            # raises error if user is using undefined signal
            model_type = model[key]
            parameter_type = kwargs[key].__class__
            if (model_type == parameter_type):
                pass
            else:
                raise(AttributeError("Wrong type for parameter " + str(key)))

        kwargs["name"] = name

        self.pipe.send(kwargs)
        self.signal_out.clear()
        self.signal_in. set()  # indicate that there is a signal
        self.signal_out.wait()  # wait for the backend to clear the signal

    def handleSignal(self, signal_dic):
        """Signal handling in the frontend
        """
        method_name = signal_dic.pop("name")
        method = getattr(self, method_name)
        method(**signal_dic)

    def sendSignal_(self, **kwargs):  # sendSignal_(name="test_out",..)
        """Outgoing signals: signals from backend to frontend
        """
        try:
            name = kwargs.pop("name")
        except KeyError:
            raise(AttributeError("Signal name missing"))

        # a dictionary: {"parameter_name" : parameter_type}
        model = self.outgoing_signal_defs[name]

        for key in kwargs:
            # raises error if user is using undefined signal
            try:
                model_type = model[key]
            except KeyError:
                print("your outgoing_signal_defs for",name,"is:", model)
                print("you requested key:", key)
                raise
            parameter_type = kwargs[key].__class__
            if (model_type == parameter_type):
                pass
            else:
                raise(AttributeError("Wrong type for parameter " + str(key)))

        kwargs["name"] = name

        self.childpipe.send(kwargs)

    # *** backend methods corresponding to each incoming signals ***

    def stop_(self):
        self.running = False

    def test_(self, test_int=0, test_str="nada"):
        print(self.pre, "test_ signal received with", test_int, test_str)

    # ** frontend methods corresponding to each incoming signal: these communicate with the backend via pipes **

    def stop(self):
        self.sendSignal(name="stop_")

    def test(self, **kwargs):
        dictionaryCheck(self.incoming_signal_defs["test_"], kwargs)
        kwargs["name"] = "test_"
        self.sendSignal(**kwargs)

    # ** frontend methods corresponding to each outgoing signal **

    # typically, there is a QThread in the frontend-side reading the process pipe
    # the QThread reads kwargs dictionary from the pipe, say
    # {"name":"test_o", "test_str":"eka", "test_int":1}
    # And calls handleSignal(kwargs)

    def test_o(self, **kwargs):
        pass


class ValkkaShmemRGBProcess(ValkkaProcess):
    """An example process, using the valkka shared memory client for RGB images
    """

    incoming_signal_defs = {  # each key corresponds to a front- and backend methods
        "test_": {"test_int": int, "test_str": str},
        "stop_": [],
        "ping_": {"message": str}
    }

    outgoing_signal_defs = {
        "pong_o": {"message": str}
    }

    parameter_defs = {
        "image_dimensions": (tuple, (1920 // 4, 1080 // 4)),
        "n_ringbuffer": (int, 10),                 # size of the ringbuffer
        "memname": str,
        "mstimeout": (int, 1000),
        "verbose": (bool, False)
    }

    def __init__(self, name, affinity=-1, **kwargs):
        super().__init__(name, affinity)
        # check kwargs agains parameter_defs, attach ok'd parameters to this
        # object as attributes
        parameterInitCheck(ValkkaShmemRGBProcess.parameter_defs, kwargs, self)
        typeCheck(self.image_dimensions[0], int)
        typeCheck(self.image_dimensions[1], int)

    def preRun_(self):
        """After the fork, but before starting the process loop
        """
        super().preRun_()
        self.client = ShmemRGBClient(
            name=self.memname,
            n_ringbuffer=self.n_ringbuffer,
            width=self.image_dimensions[0],
            height=self.image_dimensions[1],
            mstimeout=self.mstimeout,
            verbose=self.verbose
        )

    def postRun_(self):
        """Just before process exit
        """
        print(self.pre, "post: bye!")

    def cycle_(self):
        index, isize = self.client.pull()
        if (index is None):  # semaphore timed out
            print("Semaphore timeout")
        else:
            print("Current index, size=", index, isize)
            print("Payload=", self.client.shmem_list[index][0:min(isize, 10)])

    # *** backend methods corresponding to incoming signals ***

    def stop_(self):
        self.running = False

    def test_(self, test_int=0, test_str="nada"):
        print(self.pre, "test_ signal received with", test_int, test_str)

    def ping_(self, message="nada"):
        print(
            self.pre,
            "At backend: ping_ received",
            message,
            "sending it back to front")
        self.sendSignal_(name="pong_o", message=message)

    # ** frontend methods launching incoming signals

    def stop(self):
        self.sendSignal(name="stop_")

    def test(self, **kwargs):
        dictionaryCheck(self.incoming_signal_defs["test_"], kwargs)
        kwargs["name"] = "test_"
        self.sendSignal(**kwargs)

    def ping(self, **kwargs):
        dictionaryCheck(self.incoming_signal_defs["ping_"], kwargs)
        kwargs["name"] = "ping_"
        self.sendSignal(**kwargs)

    # ** frontend methods handling received outgoing signals ***
    def pong_o(self, message="nada"):
        print("At frontend: pong got message", message)


def test5():
    st = """Test ValkkaShmemRGBProcess
  """
    pre = pre_mod + "test5 :"
    print(pre, st)

    p = ValkkaShmemRGBProcess(
        "process1",
        image_dimensions=(1920 // 4, 1080 // 4),
        n_ringbuffer=10,
        memname="testing",
        mstimeout=1000,
        verbose=True
    )

    p.start()
    time.sleep(5)
    p.test(test_int=10, test_str="kikkelis")
    time.sleep(3)
    p.stop()


def test6():
    from threading import Thread

    st = """Test ValkkaShmemProcess with a thread running in the frontend
  """
    pre = pre_mod + "test6 :"
    print(pre, st)

    class FrontEndThread(Thread):

        def __init__(self, valkka_process):
            super().__init__()
            self.valkka_process = valkka_process

        def run(self):
            self.loop = True
            pipe = self.valkka_process.getPipe()
            while self.loop:
                st = pipe.recv()  # get signal from the process
                self.valkka_process.handleSignal(st)

        def stop(self):
            self.loop = False

    p = ValkkaShmemRGBProcess(
        "process1",
        image_dimensions=(1920 // 4, 1080 // 4),
        n_ringbuffer=10,
        memname="testing",
        mstimeout=1000,
        verbose=True
    )

    # thread running in front-end
    t = FrontEndThread(p)
    t.start()

    # process runs in its own virtual memory space - i.e. in the backend
    p.start()
    time.sleep(5)
    p.ping(message="<hello from front>")
    time.sleep(3)
    p.ping(message="<hello again from front>")
    time.sleep(3)
    t.stop()
    p.ping(message="<hello once again from front>")
    t.join()
    p.stop()


def test7():
    from threading import Thread
    import select

    st = """Several ValkkaShmemProcesses with a single frontend thread.
  """
    pre = pre_mod + "test7 :"
    print(pre, st)

    class FrontEndThread(Thread):

        def __init__(self, valkka_process, valkka_process2):
            super().__init__()
            self.valkka_process = valkka_process
            self.valkka_process2 = valkka_process2
            self.process_by_pipe = {
                self.valkka_process. getPipe(): self.valkka_process,
                self.valkka_process2.getPipe(): self.valkka_process2
            }

        def run(self):
            self.loop = True
            rlis = []
            for key in self.process_by_pipe:
                rlis.append(key)
            wlis = []
            elis = []
            while self.loop:
                tlis = select.select(rlis, wlis, elis)
                for pipe in tlis[0]:
                    p = self.process_by_pipe[pipe]
                    # print("receiving from",p,"with pipe",pipe)
                    st = pipe.recv()  # get signal from the process
                    # print("got from  from",p,"with pipe",pipe,":",st)
                    p.handleSignal(st)

        def stop(self):
            self.loop = False

    p = ValkkaShmemRGBProcess(
        "process1",
        image_dimensions=(1920 // 4, 1080 // 4),
        n_ringbuffer=10,
        memname="testing",
        mstimeout=1000,
        verbose=True
    )

    p2 = ValkkaShmemRGBProcess(
        "process2",
        image_dimensions=(1920 // 4, 1080 // 4),
        n_ringbuffer=10,
        memname="testing2",
        mstimeout=1000,
        verbose=True
    )

    # thread running in front-end
    t = FrontEndThread(p, p2)
    t.start()

    # process runs in its own virtual memory space - i.e. in the backend
    p. start()
    p2.start()
    time.sleep(5)

    p. ping(message="<hello from front>")
    p2.ping(message="<hello from front2>")
    time.sleep(3)

    p. ping(message="<hello again from front>")
    time.sleep(3)

    t.stop()

    p. ping(message="<hello once again from front>")
    p2.ping(message="<second hello from front2>")
    t.join()

    p.stop()
    p2.stop()


def main():
    pre = pre_mod + "main :"
    print(pre, "main: arguments: ", sys.argv)
    if (len(sys.argv) < 2):
        print(pre, "main: needs test number")
    else:
        st = "test" + str(sys.argv[1]) + "()"
        exec(st)


if (__name__ == "__main__"):
    main()
