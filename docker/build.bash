#!/bin/bash
MY_LOCAL_VALKKA_CORE_TMP_COPY="$HOME/C/valkka_docker_tmp/valkka-core"
echo $1 > $MY_LOCAL_VALKKA_CORE_TMP_COPY/LATEST_ARCH
docker run \
--device=/dev/dri:/dev/dri \
--mount type=bind,source=$MY_LOCAL_VALKKA_CORE_TMP_COPY,target=/valkka-core \
valkka:$1 \
/bin/bash -c \
"./easy_build.bash && \
make -f debian/rules build && \
cd build_dir && \
make package && \
dpkg -i Valkka-*.deb || \
apt-get -fy install && \
python3 -c 'from valkka.core import *; print(\"VALKKA_IMPORT_TEST AT BUILD\", LiveThread(\"test\"))'"
