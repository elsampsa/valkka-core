## The official Valkka docker image - available in dockerhub:
## https://hub.docker.com/r/elsampsa/valkka
##
## building:
# docker build -t valkka:latest/or-version .
##
## when running, use the following additional options:
# docker run --shm-size=2gb --device=/dev/dri:/dev/dri valkka:1.5.3 quicktest.py
## --> (hopefully) enough sharedmem for shmem infra
## --> VAAPI hw acceleration works
##
## to test interactively, run scripts/docker.bash
FROM ubuntu:22.04
USER root
COPY Valkka.deb /tmp/Valkka.deb
WORKDIR /tmp
RUN apt-get -y update && \
apt-get install -y curl && \
dpkg -i Valkka.deb || \
apt-get install -fy
RUN curl https://raw.githubusercontent.com/elsampsa/valkka-examples/master/quicktest.py -o quicktest.py
RUN python3 quicktest.py
