#!/bin/bash
docker run --mount \
    type=bind,source="$(pwd)"/../../valkka_docker_tmp/valkka-core,target=/valkka valkka:tmp-armv8 \
    /bin/bash -c "./prepare_build.bash; make -f debian/rules build; cd build_dir; make package"
