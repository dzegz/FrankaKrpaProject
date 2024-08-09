ARG UBUNTU_VERSION=18.04

FROM vicoslab/echolib:${UBUNTU_VERSION}

LABEL maintainer "domen.tabernik@fri.uni-lj.si"

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /opt

# Install dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgtk2.0-dev \
        libusb-1.0 \
        ffmpeg \
        gnupg \
        software-properties-common \
        curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install SDK
ARG UBUNTU_VERSION
COPY scripts/99-k4a.rules /etc/udev/rules.d/
COPY scripts/install_azure_kinect_camera_u_${UBUNTU_VERSION}.sh /opt/install.sh
RUN bash /opt/install.sh

# Install Python libraries
RUN apt-get install -y --no-install-recommends python3-opencv python3-setuptools python3-dev
RUN pip3 install --no-cache-dir -U pyk4a typing-extensions numpy

COPY config.py /opt/config.py
COPY camera.py /opt/camera.py
RUN chmod +x /opt/camera.py

ENTRYPOINT ["/opt/camera.py"]