FROM ubuntu:22.04

RUN \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        cmake ninja-build g++ g++-aarch64-linux-gnu g++-arm-linux-gnueabihf g++-riscv64-linux-gnu gcc-riscv64-linux-gnu ccache \
        python3

VOLUME /opencv
VOLUME /workspace

ENV OPENCV=/opencv
ENV BUILD=/workspace/build
ENV OPENCV_DOWNLOAD_PATH=/workspace/.cache/download
ENV CCACHE_PATH=/workspace/.cache/ccache

WORKDIR /workspace
