import traceback
from multiprocessing import Event
from valkka import core
"""Create a group of Events (EventGroup) and a context manager (SyncIndex) to wait
in the main python process for operation completion in the multiprocessing backend 
(i.e. "on the other side" of the fork)

At your main python process' ctor, do this:

::

    def __init__(self, ...):
        ...
        self.eg = EventGroup(20)
        ...


In your main python process (aka "multiprocessing frontend"), you can now do this:

::

    with SyncIndex(self.eg) as i:
        # send some message to the backend, communicating
        # the index i therein
        # this section waits until the event corresponding
        # to i has been set by the backend


In the other side of the fork (aka "multiprocessing backend"), do this:

::

    # backend has been given the index i
    # does the blocking operation, and after that call this:
    self.eg.set(i)

"""


class NotEnoughEvents(BaseException):
    pass


class EventGroup:

    def __init__(self, n):
        self.events = []
        self.events_index = []
        for i in range(n):
            self.events.append(Event())
            self.events_index.append(i)

    def __str__(self):
        st = "<EventGroup: "
        for i in range(len(self.events)):
            if i in self.events_index:
                st += "f"+str(i)+" "
            else:
                st += "R"+str(i)+" "
        st += ">"
        return st

    def __len__(self):
        return len(self.events)

    
    def set(self, i):
        self.events[i].set()


class SyncIndex:

    def __init__(self, event_group: EventGroup):
        self.eg = event_group
        self.i = None

    def __enter__(self):
        try:
            self.i = self.eg.events_index.pop(0) # reserve an avail event as per index
        except IndexError:
            raise(NotEnoughEvents("init your event group with more events"))
        self.eg.events[self.i].clear() # clear the event before usage
        return self.i

    def __exit__(self, type, value, tb):
        if tb:
            print("SyncIndex failed with:")
            traceback.print_tb(tb)
        self.eg.events[self.i].wait() # wait until the event has been set
        # ..typically on the multiprocess backend
        self.eg.events_index.insert(0, self.i) # recycle the event

    """
    # async part: TODO
    # for cases when we have async _frontend_
    # (that case has not yet been considered/implemented)
    # with async backend, EventGroup & SyncIndex should work as
    # in the sync backend case

    async def __aenter__(self):
         return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.quit()
    """


class EventFdGroup:
    """A group of EventFd synchronization primitives

    You should instantiate an object from this class before spawning any
    multiprocesses: this way the sync primitives are visible to all of them.
    A global "singleton" module is a good way to do this (see demo_singleton.py)

    The list if EventFds (self.events) stays constant accross multiprocesses.

    Main process (aka frontend) reserves EventFds from this list and communicates
    the list index to the multiprocess (aka backend) which then knows which EventFd
    to pick up from the list.

    Why like this?

    Because we can't send an EventFd object through the intercom pipe between the multiprocessing
    back- and frontend.  So we just send an index number.

    To summarize:

    - EventFd's are created before forking (so they're visible for everyone)
    - After fork, only list indices are communicated between the main process and 
    the multiprocess

    """
    def __init__(self, n):
        # created & cached eventfds: stays constant 
        self.events = []
        # a list of indexes of available eventfds in self.events
        self.index = []

        for i in range(n):
            self.events.append(core.EventFd())
            #
            # core.EventFd encapsulates an event file descriptor:
            #
            # https://linux.die.net/man/2/eventfd
            # 
            # EventFd has method getFd that returns the numerical value of
            # the file descriptor
            self.index.append(i)


    def __str__(self):
        st = ""
        for i, eventfd in enumerate(self.events):
            if i in self.index:
                res = ""
            else:
                res = "RESERVED"
            st += str(i) + " " + str(eventfd) + " " + res + "\n"
        return st


    def reserve(self) -> tuple:
        """Reserver an EventFd sync primitive.  Returns a tuple of

        ::

            (index, EventFd)

        Use at process frontend / python main process
        """
        try:
            index = self.index.pop(0)
        except IndexError as e:
            pass
        else:
            return index, self.events[index]
        raise(IndexError("You've run out of EventFds: create some more"))

    def release(self, eventfd: core.EventFd):
        """Release an EventFd sync primitive

        Use at process frontend / python main process
        """
        index = self.events.index(eventfd)
        self.index.append(index)

    def release_ind(self, index: int):
        """Release an EventFd sync primitive, based on the index

        Use at process frontend / python main process
        """
        self.index.append(index)

    def fromIndex(self, i):
        """Get an EventFd, based on an index number

        Use at process backend to get corresponding EventFd (as in the frontend)
        """
        return self.events[i]


def main1():
    # raise(NotEnoughEvents)
    # eg = EventGroup(0)
    eg = EventGroup(1)
    with SyncIndex(eg) as i:
        print("waiting ", i)
        print(eg)
        # kokkelis

def main2():
    # g = EventFdGroup(10)
    g = EventFdGroup(1)
    print(g)
    ind1, e1 = g.reserve()
    print(">>",ind1, e1)
    print(g)
    ind2, e2 = g.reserve()
    print(">>",ind2, e2)
    print(g)
    print("->",g.fromIndex(ind2))
    g.release(e1)
    print(g)


if __name__ == "__main__":
    # main1()
    main2()


