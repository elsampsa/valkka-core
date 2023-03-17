import traceback
from multiprocessing import Event
# from valkka import core # nopes

class NotEnoughEvents(BaseException):
    pass

class EventGroup:
    """Creates a group of multiprocessing events

    :param n: number of events to be instantiated and cached
    :param event_class: a multiprocessing event class that has ``set`` and ``clear`` methods.
                        default: python ``multiprocessing.Event``.  Can also be ``EventFd`` from libValkka.
    
    """
    def __init__(self, n = 10, event_class = Event):
        self.events = [] # list of cached events: immutable
        self.index = [] # list of indexes of available events: mutable
        for i in range(n):
            self.events.append(event_class()) 
            self.index.append(i) 

    def __str__(self):
        st = "<EventGroup: "
        for i in range(len(self.events)):
            if i in self.index:
                st += "f"+str(i)+" "
            else:
                st += "R"+str(i)+" "
        st += ">"
        return st

    def __len__(self):
        return len(self.events)
    
    def set(self, i):
        """Set / trigger an event at index i.  Used typically at multiprocessing backend.
        """
        self.events[i].set()

    def reserve(self) -> tuple:
        """Reserve and return an Event instance together with its index: ``index, Event``

        Use typically at process frontend / python main process
        """
        try:
            index = self.index.pop(0)
        except IndexError as e:
            raise NotEnoughEvents
        event = self.events[index]
        event.clear() # clear event before using it
        return index, event
        
    def release(self, event):
        """Release an EventFd sync primitive.  
        Use typically at process frontend / python main process

        :param event: event to be released / returned
        """
        try:
            index = self.events.index(event)
        except ValueError: # trying to return an event that's not in this EventGroup
            raise ValueError("event not in this Eventgroup")
        self.index.append(index)

    def release_ind(self, index: int):
        """Release an EventFd sync primitive, based on the index.
        Use typically at process frontend / python main process

        :param index: event's index
        """
        self.index.append(index)

    def fromIndex(self, i):
        """Get an event, based on the event index.
        Use typically at multiprocessing backend to get the corresponding event as in the frontend.
        """
        return self.events[i]

    def asIndex(self, event):
        """Return index corresponding to an event
        """
        return self.events.index(event)


class SyncIndex:
    """A context manager for synchronizing between multiprocessing front- and backend.

    :param event_group: an EventGroup instance

    Wait's and releases an event at context manager exit
    """
    def __init__(self, event_group: EventGroup):
        self.eg = event_group
        self.event = None

    def __enter__(self):
        i, self.event = self.eg.reserve()
        return i

    def __exit__(self, type, value, tb):
        if tb:
            print("SyncIndex failed with:")
            traceback.print_tb(tb)
        self.event.wait() # wait until the event has been set
        self.eg.release(self.event) # recycle the event
    


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


