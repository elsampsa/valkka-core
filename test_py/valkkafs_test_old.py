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


def test2():
    from valkka.api2 import findBlockDevices
    devs=findBlockDevices()
    print(devs)


def test3():
    from valkka.api2 import ValkkaFS
    print("test3")
    fs = ValkkaFS.newFromDirectory(dirname="/home/sampsa/tmp/testvalkkafs", blocksize=1024*10, n_blocks=10)
    fs.report()
    
    
def test4():
    from valkka.api2 import ValkkaFS
    print("test4")
    fs = ValkkaFS.loadFromDirectory(dirname="/home/sampsa/tmp/testvalkkafs")
    fs.report()
    bt = fs.getBlockTable()
    print(bt[:,0:10])


def test5():
    from valkka.api2 import ValkkaFS
    print("test5")
    # WARNING: will wipe out /dev/sda
    return
    # fs = ValkkaFS.newFromDirectory(dirname="/home/sampsa/tmp/testvalkkafs", dumpfile="/dev/sda", blocksize=20*1024*1024, n_blocks=3)
    fs = ValkkaFS.newFromDirectory(dirname="/home/sampsa/tmp/testvalkkafs", dumpfile="/dev/sda", blocksize=20*1024*1024, n_blocks=23370)
    fs.report()
    
    
def test6():
    from valkka.api2 import ValkkaFS
    # blocktable with write-edge
    fs = ValkkaFS.newFromDirectory(dirname="/home/sampsa/tmp/testvalkkafs", blocksize=1024*10, n_blocks=10)
    # number of blocks = 10
  
    """
    void markFrame(long int mstimestamp);        // <pyapi>
    void markKeyFrame(long int mstimestamp);     // <pyapi>
    """
    
    for i in range(0,22):
        # first i frames
        num=(i+1)*100
        num2=(i+2)*100
        if (i<=4): # simulate a write edge
            num=(i+1+20)*100
            num2=(i+2+20)*100
        fs.core.markFrame(num)
        fs.core.markKeyFrame(num)
        fs.core.markKeyFrame(num2+3)
        # second i frames
        # bt.markKeyFrame(1,num+2)
        # bt.markKeyFrame(2,num2+3)
        fs.core.writeBlock(False)


    bt = fs.getBlockTable()
    for i, row in enumerate(bt):
        print(i, row)
    
    """
    [[2203 2203]
    [2303 2303]
    [   0    0]
    [1503 1503]
    [1603 1603]
    [1703 1703]
    [1803 1803]
    [1903 1903]
    [2003 2003]
    [2103 2103]]
    """

    print()
    req = (1720, 2210)
    print(req)
    print(fs.getInd(req))

    print()
    req = (1720, 2400)
    print(req)
    print(fs.getInd(req))

    print()
    req = 1700
    print(req)
    print(fs.getIndNeigh(n=2, time=req))

    print()
    req = 2300
    print(req)
    print(fs.getIndNeigh(n=2, time=req))


def test7():
    from valkka.api2 import ValkkaFS
    fs = ValkkaFS.newFromDirectory(dirname="/home/sampsa/tmp/testvalkkafs", blocksize=1024*10, n_blocks=10)
    
    nn=1
    for i in range(0,nn):
        # first i frames
        num=(i+1)*100
        num2=(i+2)*100
        if (i<=4): # simulate a write edge
            num=(i+1+20)*100
            num2=(i+2+20)*100
        fs.core.markFrame(num)
        fs.core.markKeyFrame(num)
        fs.core.markKeyFrame(num2+3)
        # second i frames
        # fs.core.markKeyFrame(1,num+2)
        # fs.core.markKeyFrame(2,num2+3)
        fs.core.writeBlock()
        
    bt = fs.getBlockTable()
    for i, row in enumerate(bt):
        print(i, row)
        
    print()
    req = (1720, 2210)
    print(req)
    print(fs.getInd(req))

    print()
    req = (1720, 2400)
    print(req)
    print(fs.getInd(req))

    print()
    req = 1700
    print(req)
    print(fs.getIndNeigh(n=2, time=req))

    print()
    req = 2300
    print(req)
    print(fs.getIndNeigh(n=2, time=req))


def test8():
    from valkka.api2 import ValkkaFS
    fs = ValkkaFS.newFromDirectory(dirname="/home/sampsa/tmp/testvalkkafs", blocksize=1024*10, n_blocks=10)

    nn=22
    # fs.core.markKeyFrame(90)
    for i in range(0,nn):
        # first i frames
        num=(i+1)*100
        num2=(i+2)*100
        if (i<=4): # simulate a write edge
            num=(i+1+20)*100
            num2=(i+2+20)*100
        fs.core.markFrame(num)
        # fs.core.markKeyFrame(num)
        # fs.core.markKeyFrame(num2+3)
        # second i frames
        # fs.core.markKeyFrame(1,num+2)
        # fs.core.markKeyFrame(2,num2+3)
        fs.core.writeBlock()

    bt = fs.getBlockTable()
    for i, row in enumerate(bt):
        print(i, row)
    
    print()
    req = (1720, 2210)
    print(req)
    print(fs.getInd(req))

    print()
    req = (1720, 2400)
    print(req)
    print(fs.getInd(req))

    print()
    req = 1700
    print(req)
    print(fs.getIndNeigh(n=2, time=req))

    print()
    req = 2300
    print(req)
    print(fs.getIndNeigh(n=2, time=req))

     

if (__name__=="__main__"):
    st="test"+str(sys.argv[1])+"()"
    exec(st)

    




