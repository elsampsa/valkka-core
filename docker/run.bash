#!/bin/bash
MY_LOCAL_VALKKA_CORE_TMP_COPY="$HOME/C/valkka_docker_tmp/valkka-core"
docker run \
--device=/dev/dri:/dev/dri \
--mount type=bind,source=$MY_LOCAL_VALKKA_CORE_TMP_COPY,target=/valkka-core \
-it valkka:$1 /bin/bash
