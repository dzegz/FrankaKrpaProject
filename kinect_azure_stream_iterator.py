import echolib
import torch
from torchvision import transforms
from echolib.camera import FrameSubscriber

class KinectAzureStreamIterator:
    def __init__(self):
        self.color_frame = None
        self.depth_frame = None

        self.loop = echolib.IOLoop()
        client = echolib.Client()
        self.loop.add_handler(client)

        self.color_sub = FrameSubscriber(client, "azure_kinect_rgb", self.rec_color)
        self.depth_sub = FrameSubscriber(client, "azure_kinect_depth", self.rec_depth)

        print("KinectAzureStreamIterator created")

    def __iter__(self):
        return self

    def rec_color(self, x):
        self.color_frame = x.image

    def rec_depth(self, x):
        self.depth_frame = x.image

    def __next__(self):
        while self.loop.wait(10):
            if self.color_frame is not None and self.depth_frame is not None:
                color_return = self.color_frame
                depth_return = self.depth_frame

                self.color_frame = None
                self.depth_frame = None
                
                return {"image": color_return, "depth": depth_return}

if __name__ == '__main__':
    import cv2
    from PIL import Image

    it = KinectAzureStreamIterator()

    for i, s in enumerate(it):
        print(s)
        