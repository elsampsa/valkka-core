#!/bin/bash
cp config.linux-generic live/
cd live
## this doesn't work at PPA's autobuild:
#arch=$(uname --m)
arch=$1
echo "CONFIGURE_LIVE: ARCH "$arch
if [[ $arch = "x86_64" || $arch = "amd64" ]];
then
    echo "LIVE555: using linux-64bit"
    ./genMakefiles linux-64bit
elif [[ $arch = *"arm"* ]];
then
    echo "LIVE555: using armlinux"
    ./genMakefiles armlinux
else
    echo "LIVE555: WARNING: using generic linux"
    ./genMakefiles linux-generic
fi
## so we'll skip it & don't build for arm for the moment..
#./genMakefiles linux-64bit
cd ..
