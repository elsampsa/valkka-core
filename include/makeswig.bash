#!/bin/bash
#
# extract everything that we want to expose in python
#

headers="thread.h framefifo.h framefilter.h threadsignal.h livethread.h filethread.h fileframefilter.h avthread.h openglthread.h openglframefifo.h sharedmem.h logging.h constant.h avdep.h testthread.h framefilterset.h"

for header in $headers
do
  grep -h "<pyapi>" $header | awk '{if ($1=="class" || $1=="enum" || $1=="struct") {print " "}; print $0}'
done
