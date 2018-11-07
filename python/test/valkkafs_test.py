import sys
import numpy
from valkka.core import ValkkaFS

def test1():
    """Test basic ValkkaFS book-keeping
    """

    def callback(arg):
        print("callback :", arg)

    v = ValkkaFS("eka", "toka", 100, 10)
    v.setBlockCallback(callback)

    a = numpy.zeros((v.get_n_blocks(), v.get_n_cols()),dtype=numpy.int_)

    v.setArrayCall(a)
    print(a)

    for i in range(100+1):
        ts = i*1000;
        if (i%10==0):
            v.markKeyFrame(ts)
        else:
            v.markFrame(ts)

    v.writeBlock()

    v.setArrayCall(a)
    print(a)

    v.writeBlock()
    v.setArrayCall(a)
    print(a)

    for j in range(2,20): # blocks
        for i in range(100+1):
            ts = j*i*1000;
            if (i%10==0):
                v.markKeyFrame(ts)
            else:
                v.markFrame(ts)
        v.writeBlock()
        v.setArrayCall(a)
        print(a)


if (__name__=="__main__"):
    st="test"+str(sys.argv[1])+"()"
    exec(st)

    




