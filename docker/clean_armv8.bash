#!/bin/bash
docker run --mount type=bind,source="$(pwd)"/../../valkka_docker_tmp,target=/valkka valkka:tmp-armv8 /bin/bash -c "make -f debian/rules clean"
