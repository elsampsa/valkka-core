#!/bin/bash
## WARNING: if build fails, use with --no-cache
##
## why we don't want to make a copy of the code in here?
## we don't want it into a docker context & build an image with it
## reasons: see the explanation in the Dockerfile(s)
#echo
#echo "creating a local copy of libValkka here using rsync"
#echo
# rm -rf valkka_tmp
#mkdir -p valkka_tmp
# ls .. -1 | grep -v "docker"  | grep -v "build_dir" | xargs -i cp -r ../{} valkka_tmp/
#rsync -r -u --delete --max-size=1M \
#--exclude 'debug/' \
#--exclude '__pycache__' \
#--exclude '*.o' \
#--exclude '*.so' \
#--exclude '*.so.*' \
#--exclude '*.a' \
#--exclude '*.tar.gz' \
#--exclude 'docker' \
#--exclude 'build_dir' \
#--exclude '*.deb' \
#--exclude 'aux/' \
#../* valkka_tmp/
#
## for armv8:
if [ "$1" == "armv8_ubuntu_18" ]
then
    cp -f /usr/bin/qemu-arm-static ./
fi
#
echo
# echo "docker build & compile"
echo "docker build"
echo 
docker build . -f Dockerfile.$1 -t valkka:$1 $2
