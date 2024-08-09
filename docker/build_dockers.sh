#!/bin/bash

MODEL_VERSION="depth_large"

echo Building Kinect2 image
docker build -t camera-kinect2 camera
docker build -t camera-kinect_azure camera_azure

echo Building detector image

# download main model if does no exist
if [ ! -f detector/${MODEL_VERSION}.pth ]; then
    wget -O detector/${MODEL_VERSION}.pth https://box.vicos.si/vicos-cube/cloth/docker/${MODEL_VERSION}.pth
fi

# download center model if does no exist
if [ ! -f detector/center_model.pth ]; then
    wget -O detector/center_model.pth https://box.vicos.si/vicos-cube/cloth/docker/center_model.pth
fi

docker build \
    -t cloth-corner-detector \
    --build-arg CUDA_VERSION=11.8.0-cudnn8 \
    --build-arg CEDIRNET_VERSION=$(date +%s) \
    --build-arg MODEL_VERSION=${MODEL_VERSION} \
    detector
