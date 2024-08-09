import sys
import cv2
import echolib
from echolib.camera import FrameSubscriber

def display(frame):
    canvas = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
    cv2.putText(canvas, str(frame.header.timestamp), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
    cv2.imshow("Image", canvas)
    cv2.waitKey(2)

def main():

    class FrameCollector(object):

        def __init__(self):
            self.frame = None

        def __call__(self, x):
            self.frame = x

    collector = FrameCollector()

    loop = echolib.IOLoop()
    client = echolib.Client()
    loop.add_handler(client)

    sub = FrameSubscriber(client, "azure_kinect_rgb", collector)

    try:
        while loop.wait(10):
            if collector.frame is not None:
                display(collector.frame)
                collector.frame = None

    except KeyboardInterrupt:
        pass

    sys.exit(1)

if __name__ == '__main__':
    main()