#!/bin/bash

# Determine the target architecture
ARCH=$(dpkg-architecture -qDEB_HOST_ARCH)
echo
echo Producing debian control file for architecture $ARCH
echo

build_deps="build-essential, yasm, cmake, pkg-config, swig, libglew-dev, mesa-common-dev, python3-dev, python3-numpy, libasound2-dev, libssl-dev, coreutils, freeglut3-dev"
vaapi_build_deps="i965-va-driver, libva-dev"

deps="python3, mesa-utils, glew-utils, python3-numpy, v4l-utils, python3-pip, openssl, arp-scan"
vaapi_deps="i965-va-driver, intel-gpu-tools, vainfo"

if [[ ! $arch = *"arm"* ]]; then
    echo "control.sh: enabling VAAPI"
    build_deps=$build_deps", "$vaapi_build_deps
    deps=$deps", "$vaapi_deps
else
    echo "control.sh: disabling VAAPI for arm architecture"
fi

build="\
Source: valkka
Section: libs
Priority: optional
Maintainer: Sampsa Riikonen <sampsa.riikonen@iki.fi>
Homepage: https://elsampsa.github.io/valkka-examples/_build/html/index.html
Build-Depends: "$build_deps"\n"

pkg="\
Package: valkka
Architecture: any
Description: Valkka
 OpenSource Video Management for Linux
Depends: "$deps"\n"

# or better?
# Architecture: "$ARCH" \n

target="control"
echo -e "$build" > "$target"
echo -e "$pkg" >> "$target"
