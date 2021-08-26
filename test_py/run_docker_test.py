#!/bin/bash
# image="ubuntu:18.04"
image="elsampsa/valkka:ubuntu18-src-1.2.0"
# command="/tmp/pipe_test.py"
command="/tmp/aux.py"
docker run -e PYTHONUNBUFFERED=1 --mount type=bind,source="$(pwd)",target=/tmp $image python3 $command

