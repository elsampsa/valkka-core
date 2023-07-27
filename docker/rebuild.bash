#!/bin/bash
MY_LOCAL_VALKKA_CORE_TMP_COPY="$HOME/C/valkka_docker_tmp/valkka-core"
# echo $1 > $MY_LOCAL_VALKKA_CORE_TMP_COPY/LATEST_ARCH # TODO: CHECK
docker run \
--device=/dev/dri:/dev/dri \
--mount type=bind,source=$MY_LOCAL_VALKKA_CORE_TMP_COPY,target=/valkka-core \
valkka:$1 \
/bin/bash -c \
"make -f debian/rules build && \
cd build_dir && \
make package && \
dpkg -i Valkka-*.deb || \
apt-get -fy install && \
vainfo && \
python3 -c 'from valkka.core import *; print(\"VALKKA_IMPORT_TEST AT REBUILD\", LiveThread(\"test\"))'"
