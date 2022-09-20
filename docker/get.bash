#!/bin/bash
target=$HOME/C/valkka_docker_tmp/valkka-core
echo $target
# sudo rm -rf $target
mkdir -p $target
rsync -v -r -u --delete --max-size=1M \
--exclude 'debug/' \
--exclude '__pycache__' \
--exclude '*.o' \
--exclude '*.so' \
--exclude '*.so.*' \
--exclude '*.a' \
--exclude '*.tar.gz' \
--exclude 'docker' \
--exclude 'build_dir' \
--exclude '*.deb' \
--exclude 'aux/' \
$HOME/C/valkka/* $target/
