#!/bin/bash
docker run --mount type=bind,source="$(pwd)"/../../valkka_docker_tmp/valkka-core,target=/valkka valkka:tmp-x86 /bin/bash -c "make -f debian/rules build; cd build_dir; make package"
