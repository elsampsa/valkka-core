#!/bin/bash
MY_LOCAL_VALKKA_CORE_TMP_COPY="$HOME/C/valkka_docker_tmp/valkka-core"
# echo $1 > $MY_LOCAL_VALKKA_CORE_TMP_COPY/LATEST_ARCH # TODO: CHECK
basename=$(head -1 Dockerfile.$1 | awk '{print $2}')
echo $basename
docker run \
--mount type=bind,source=$MY_LOCAL_VALKKA_CORE_TMP_COPY,target=/valkka-core \
-it $basename /bin/bash
