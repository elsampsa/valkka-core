#!/bin/bash
## extract base image name from the dockerfiles
MY_LOCAL_VALKKA_CORE_TMP_COPY="$HOME/C/valkka_docker_tmp/valkka-core"
# echo $1 > $MY_LOCAL_VALKKA_CORE_TMP_COPY/LATEST_ARCH # TODO: CHECK
basename=$(head -1 Dockerfile.$1 | awk '{print $2}')
echo $basename
docker run \
--mount type=bind,source=$MY_LOCAL_VALKKA_CORE_TMP_COPY,target=/valkka-core \
$basename \
/bin/bash -c "cd /valkka-core/build_dir && \
apt-get update && \
dpkg -i Valkka-*.deb || \
apt-get -fy install && \
python3 -c 'from valkka.core import *; print(\"VALKKA_IMPORT_TEST AT TEST\", LiveThread(\"test\"))'"
# needs "apt-get update" since this is just a very empty image..
