ARG UBUNTU_VERSION=20.04

FROM vicoslab/echolib:${UBUNTU_VERSION}

LABEL maintainer "domen.tabernik@fri.uni-lj.si"

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /opt

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget pkg-config python3-venv \
        libusb-1.0-0-dev libturbojpeg0-dev libglfw3-dev \
        ffmpeg libjpeg-dev libopenni2-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && python3 -m pip install setuptools
RUN python3 -m pip install opencv_python_headless cython

RUN mkdir /etc/udev
RUN mkdir /etc/udev/rules.d

# Install libfreenect2
RUN git clone https://github.com/OpenKinect/libfreenect2.git /opt/libfreenect2
WORKDIR /opt/libfreenect2
RUN mkdir mkdir build && cd build && cmake .. && make -j install

# install udev rules for kinect2
RUN cp /opt/libfreenect2/platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/

# install python bindings for libfreenect2
RUN apt update && apt install python3-dev python3-pip -y && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install wheel && PKG_CONFIG_PATH=/usr/lib/pkgconfig python3 -m pip install freenect2 

# Fixing some problems with python kinect binding library
WORKDIR /opt
RUN ln -s /usr/local/lib/python3.8/dist-packages/freenect2/_freenect2.abi3.so
COPY fixed_library.py /usr/local/lib/python3.8/dist-packages/freenect2/__init__.py
    
COPY camera.py /opt/camera.py
RUN chmod +x /opt/camera.py

ENTRYPOINT ["/opt/camera.py"]
