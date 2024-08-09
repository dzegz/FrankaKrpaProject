#!/usr/bin/env python3
import echolib
import cv2
import numpy as np
import signal
from freenect2 import Device, FrameType
from freenect2 import Frame as KinectFrame
from echolib.camera import FramePublisher, Frame

from _freenect2 import lib, ffi


class FrameCollector:
    def __init__(self):
        self.rgb = None
        self.depth = None

        self.loop = echolib.IOLoop()
        client = echolib.Client()
        self.loop.add_handler(client)

        self.output_rgb = FramePublisher(client, "kinect_rgb")
        self.output_depth = FramePublisher(client, "kinect_depth")

    def set_registration(self, regist):
        self.regist = regist

        self.undistorted = KinectFrame.create(512, 424, 4)
        self.registered = KinectFrame.create(512, 424, 4)
        self.big_depth = KinectFrame.create(1920, 1082, 4)

    # Function from the library, trying to fix a memory leak
    def register(self, rgb, depth, enable_filter=True, with_big_depth=False):

        self.undistorted.format = depth.format
        self.registered.format = rgb.format

        if with_big_depth:
            self.big_depth.format = depth.format
            big_depth_ref = self.big_depth._c_object
        else:
            big_depth, big_depth_ref = None, ffi.NULL

        lib.freenect2_registration_apply(
            self.regist._c_object,
            rgb._c_object, depth._c_object, self.undistorted._c_object,
            self.registered._c_object, 1 if enable_filter else 0,
            big_depth_ref
        )

        rvs = [self.undistorted, self.registered]
        if with_big_depth:
            rvs.append(self.big_depth)

        return tuple(rvs)

    def add_rgb(self, rgb):
        self.rgb = rgb
        if self.depth is not None:
            self.send()
    
    def add_depth(self, depth):
        self.depth = depth
        if self.rgb is not None:
            self.send()

    def send(self):
        reg_depth, reg_rgb, big_depth = self.register(self.rgb, self.depth, enable_filter=True, with_big_depth=True)
        req_rgb_img = np.array(reg_rgb.to_array())
        reg_depth_img = np.array(reg_depth.to_array())

        rgb_img = np.array(self.rgb.to_array())
        big_depth = np.array(big_depth.to_array()[1:-1])
        
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        req_rgb_img = cv2.cvtColor(req_rgb_img, cv2.COLOR_BGR2RGB)

        self.loop.wait(10)

        self.output_rgb.send(Frame(image=rgb_img))
        self.output_depth.send(Frame(image=big_depth))
        self.rgb = None
        self.depth = None


device = Device()
collector = FrameCollector()

def stop(*args):
    device.stop()
    exit()

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

with device.running():
    collector.set_registration(device.registration)
    for type_, frame in device:
        if type_ is FrameType.Color:
            collector.add_rgb(frame)
            
        if type_ is FrameType.Depth:
            collector.add_depth(frame)

