#!/bin/bash
## extract base image name from the dockerfiles
MY_LOCAL_VALKKA_CORE_TMP_COPY="$HOME/C/valkka_docker_tmp/valkka-core"
# echo $1 > $MY_LOCAL_VALKKA_CORE_TMP_COPY/LATEST_ARCH # TODO: CHECK
basename=$(head -1 Dockerfile.$1 | awk '{print $2}')
echo $basename
docker run \
-it \
--device=/dev/dri:/dev/dri \
--mount type=bind,source=$MY_LOCAL_VALKKA_CORE_TMP_COPY,target=/valkka-core \
$basename \
/bin/bash -c "export TZ=Europe/Helsinki && \
ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
cd /valkka-core/build_dir && \
apt-get update && \
dpkg -i Valkka-*.deb || \
apt-get -fy install && \
vainfo && \
python3 -c 'from valkka.core import *; print(\"VALKKA_IMPORT_TEST AT TEST\", LiveThread(\"test\"))' && \
/bin/bash"
# needs "apt-get update" since this is just a very empty image..
