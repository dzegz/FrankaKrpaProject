#!/bin/bash

# UPDATE THIS to the path of the virtual environment
VIRTENV_ECHOLIB=${VIRTENV_ECHOLIB:-"/home/vicosdemo/demo"}


source $VIRTENV_ECHOLIB/bin/activate
echo Starting echolib
python3 -u -m echolib &
pid=$!
echo Echolib started

xhost local:docker

docker run -it --rm \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp/echo.sock:/tmp/echo.sock \
    --device /dev/bus/usb:/dev/bus/usb \
    -e DISPLAY="$DISPLAY" \
    camera-kinect2 

kill $pid
