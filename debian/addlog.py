#!/usr/bin/python3


"""valkka (0.13.2-0ubuntu1-xenialppa1) xenial; urgency=low

  * Fixed sending initial SPS & PPS packets from the rtsp setup string

 -- Sampsa Riikonen <sampsa.riikonen@iki.fi>  Fri, 16 Aug 2019 10:18:29 +0300
"""

"""valkka (0.13.2-0ubuntu1-bionicppa1) bionic; urgency=low

  * Fixed sending initial SPS & PPS packets from the rtsp setup string

 -- Sampsa Riikonen <sampsa.riikonen@iki.fi>  Fri, 16 Aug 2019 10:18:29 +0300
"""

import os
import sys
import subprocess


def write(version, distro, comment):
    s = subprocess.Popen("date -R".split(), stdout = subprocess.PIPE)
    stdout, stderr = s.communicate()
    date = stdout.decode("utf-8")[:-1]
    
    st = """valkka ({VERSION}-0ubuntu1-{DISTRO}ppa1) {DISTRO}; urgency=low

  * {COMMENT}

 -- Sampsa Riikonen <sampsa.riikonen@iki.fi>  {DATE}

""".format(
    VERSION = version,
    DISTRO  = distro,
    COMMENT = comment,
    DATE = date)

    print(">>>\n"+st+"\n<<<<")
    # return
    # do_write = True
    do_write = False
    
    if do_write:
        f = open("changelog", "r")
        dump = f.read()
        f.close()
        f = open("changelog", "w")
        f.write(st)
        f.write(dump)
        f.close()

    print("\nWriting 'rundeb.bash' for easy building & uploading\n")
    fname = "valkka_{VERSION}-0ubuntu1-{DISTRO}ppa1_source.changes".format(VERSION = version, DISTRO = distro)
    st = """#!/bin/bash
clear; debuild -S -sa
DIRTMP=$PWD
cd ../..
dput ppa:sampsa-riikonen/valkka %s
cd $DIRTMP
""" % fname

    with open("rundeb.bash", "w") as f:
        f.write(st)
    
    os.system("chmod a+x rundeb.bash")

    

if __name__ == "__main__":
    
    st="""Needs: version disto-name comment[optional]

Typical usecase:

    addlog.py 0.13.2 bionic new awesome version
    addlog.py 0.13.2 focal
"""
    print(st)

    if len(sys.argv) < 3:
        print("missing parameters")
        sys.exit(3)
    
    version = sys.argv[1]
    distro = sys.argv[2]
    if len(sys.argv) < 4:
        print("loading previous comment")
        comment = None
    else:
        # print(">",sys.argv[3:])
        comment = " ".join(sys.argv[3:])
    
    if comment is not None:
        with open("comment.tmp", "w") as f:
            f.write(comment)
    
    with open("comment.tmp", "r") as f:
        comment = f.read()
    
    print("summary: version:", version, "distro:", distro, "comment:", comment)
    write(version, distro, comment)


