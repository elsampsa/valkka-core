#!/bin/bash
#
# extract everything that we want to expose in python
#


# Initialize a variable to store the option value
vaapi_option=""

# thanks chatgpt!
# Function to check if an option starts with a specific string
startsWith() {
  case $1 in
    $2*) return 0;;
    *) return 1;;
  esac
}
# Loop through command-line arguments
while [ $# -gt 0 ]; do
  case "$1" in
    -vaapi)
      shift
      vaapi_option="true"
      ;;
    *)
      echo "Usage: makeswig.bash -vaapi <vaapi_value>"
      exit 1
      ;;
  esac
  shift
done

echo
echo GENERATING valkka_core.i
echo

headers="frame.h thread.h framefifo.h framefilter.h \
threadsignal.h livethread.h avfilethread.h fileframefilter.h \
decoderthread.h avthread.h openglthread.h openglframefifo.h sharedmem.h logging.h \
constant.h avdep.h testthread.h framefilterset.h filestream.h cachestream.h \
valkkafs.h usbthread.h valkkafsreader.h movement.h fdwritethread.h metadata.h \
muxer.h muxshmem.h framefilter2.h"

# Check if the option is set
if [ -n "$vaapi_option" ]; then
  echo "makeswig: using vaapi"
    headers=$headers" vaapithread.h"
else
  echo "makeswig: disabling vaapi"
fi

# init valkka_core.i
cat valkka.i.base > valkka_core.i

for header in $headers
do
    echo "// "$header >> valkka_core.i
    grep -h "<pyapi>" $header | awk '{if ($1=="class" || $1=="enum" || $1=="struct") {print " "}; print $0}' >> valkka_core.i
done

