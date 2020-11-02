#!/bin/bash
## WARNING: if build fails, use --no-cache
cp /usr/bin/qemu-arm-static ./
docker build . -f Dockerfile.armv8 -t valkka:tmp-armv8
