#!/bin/bash
#
# extract everything that we want to expose in python
#

echo
echo GENERATING valkka_core.i
echo

headers="frame.h thread.h framefifo.h framefilter.h \
threadsignal.h livethread.h avfilethread.h fileframefilter.h \
decoderthread.h avthread.h openglthread.h openglframefifo.h sharedmem.h logging.h \
constant.h avdep.h testthread.h framefilterset.h filestream.h cachestream.h \
valkkafs.h usbthread.h valkkafsreader.h movement.h fdwritethread.h metadata.h \
muxer.h muxshmem.h framefilter2.h vaapithread.h"

# init valkka_core.i
cat valkka.i.base > valkka_core.i

for header in $headers
do
    echo "// "$header >> valkka_core.i
    grep -h "<pyapi>" $header | awk '{if ($1=="class" || $1=="enum" || $1=="struct") {print " "}; print $0}' >> valkka_core.i
done

