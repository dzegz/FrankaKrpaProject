#!/usr/bin/env python3
import pyk4a
import signal
import echolib
import cv2
import numpy as np
from echolib.camera import FramePublisher, Frame
from echolib.array import TensorPublisher
from config import config

def main():
    print("Starting echolib")
    loop = echolib.IOLoop()
    client = echolib.Client()
    loop.add_handler(client)

    output_rgb = FramePublisher(client, "azure_kinect_rgb")
    output_depth = FramePublisher(client, "azure_kinect_depth")
    output_acceleration = TensorPublisher(client, "azure_kinect_acceleration")
    output_gyroscope = TensorPublisher(client, "azure_kinect_gyroscope")

    print("Opening Azure Kinect")
    k4a = pyk4a.PyK4A(config)
    k4a.start()

    def stop(*args):
        print("Exiting")
        k4a.stop()
        exit()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print("Starting streaming")
    while loop.wait(10):
        capture = k4a.get_capture()
        imu = k4a.get_imu_sample()
        if capture.transformed_depth is not None and capture.color is not None:
            rgb_image = cv2.cvtColor(capture.color, cv2.COLOR_BGR2RGB)
            output_rgb.send(Frame(image=np.array(rgb_image)))
            output_depth.send(Frame(image=np.array(capture.transformed_depth)))
            output_acceleration.send(np.array(imu.pop("acc_sample")))
            output_gyroscope.send(np.array(imu.pop("gyro_sample")))

    stop()

if __name__ == "__main__":
    main()