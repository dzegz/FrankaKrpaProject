#!/bin/bash

# UPDATE THIS to the path of the virtual environment
VIRTENV_ECHOLIB=${VIRTENV_ECHOLIB:-"/home/vicosdemo/demo"}

ECHO="/tmp/echo.sock"
CAMERACONFIG="./camera_azure/config.py"

source $VIRTENV_ECHOLIB/bin/activate
echo Starting echolib
python3 -u -m echolib &
pid=$!
echo Echolib started

xhost local:docker

docker run -it \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ${CAMERACONFIG}:/opt/config.py \
    -v ${ECHO}:${ECHO} \
    --device /dev/bus/usb:/dev/bus/usb \
    --gpus 'all,"capabilities=compute,utility,graphics"' \
    camera-kinect_azure



kill $pid
