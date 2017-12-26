#!/bin/bash
#
# extract everything that we want to expose in python
#

headers="frames.h filters.h queues.h threads.h livethread.h avthread.h openglthread.h sizes.h sharedmem.h"

for header in $headers
do
  grep -h "<pyapi>" $header | awk '{if ($1=="class" || $1=="enum" || $1=="struct") {print " "}; print $0}'
done
