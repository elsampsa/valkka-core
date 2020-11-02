#!/bin/bash
## prepare your system for multiarch docker builds
sudo apt-get update 
sudo apt-get install -y qemu binfmt-support qemu-user-static
sudo update-binfmts --enable qemu-arm
sudo update-binfmts --display qemu-arm 
# https://www.stereolabs.com/docs/docker/building-arm-container-on-x86/
## this is everything needed..?
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
## after that, no need for all this..?
# https://matchboxdorry.gitbooks.io/matchboxblog/content/blogs/build_and_run_arm_images.html
echo
echo TESTING THE EMULATION ENVIRONMENT
echo
docker run --rm -t arm64v8/ubuntu uname -m
echo
