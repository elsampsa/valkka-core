#!/bin/bash
mkdir -p docker_tmp
cp Dockerfile docker_tmp/
cd docker_tmp
VER="1.5.3"
URL="https://github.com/elsampsa/valkka-core/releases/download/"$VER"/Valkka-"$VER"-Ubuntu22.deb"
curl -L $URL -o Valkka.deb
docker build -t valkka:$VER . $@
cd ..
