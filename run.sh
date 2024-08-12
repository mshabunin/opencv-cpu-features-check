#!/bin/bash

set -exu

mkdir -p workspace

tag=opencv-cpu-config-checker
docker build -t ${tag} -f Dockerfile .
docker run -it \
    -u $(id -u):$(id -g) \
    -v `pwd`/../opencv:/opencv:ro \
    -v `pwd`/scripts:/scripts:ro \
    -v `pwd`/workspace:/workspace \
    ${tag} \
    python3 /scripts/test.py
    # -k x86 # <--- test filter
    # /bin/bash
