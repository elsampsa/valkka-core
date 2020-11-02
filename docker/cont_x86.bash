#!/bin/bash
## WARNING: if build fails, use --no-cache
docker build . -f Dockerfile.x86 -t valkka:tmp-x86
