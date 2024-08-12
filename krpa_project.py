import numpy as np
#from icecream import ic
import time
import sys

sys.path.insert(1, '/home/peter/robotblockset_python')
from robotblockset_python.panda_ros import panda_ros
from robotblockset_python.ros.grippers_ros import PandaGripper
from robotblockset_python.transformations import rot_z
from robotblockset_python.paths import trajectories

# for pixel to world coords
import csv

###############################################################################################

def main():
    #corners = get_world_coords()
    
    ####

    r = panda_ros('pingvin', init_node=True, init_frankadesk_gripper_TCP=True, start_controller='position_joint_trajectory_controller')
    time.sleep(1)
    g = PandaGripper(robot=r, namespace='pingvin')

    # going to the initial position
    q_home= np.array([0, -0.2, 0, -1.5, 0, 1.5, 0.7854]) # home joint configuration
    r.JMove(q_home, 10)

    # go out of the camera frame
    pose = np.array([
        [ 0.97199178,  0.12902535, -0.19642922,  0.06418075],
        [ 0.23329223, -0.63073377,  0.74010111,  0.31493582],
        [-0.02840274, -0.7651976 , -0.64316865,  0.47720053],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])
    r.CMove(pose, 10)

    # at this moment you should take a picture and run the pddl planning procedure
    time.sleep(3)

    # going back to the initial position
    r.JMove(q_home, 10)
    #going to the center of the coordinate system
    r.CMoveFor([0,0.11,0], 5)

    # gripper homing
    g.homing()
    g.open()

    # going down
    r.CMoveFor([0,0,-0.6], 10)
    time.sleep(1)
    r.CMoveFor([0,0,-0.025], 3)
    time.sleep(1)

    # grabbing
    g.close()

    # going up and releasing
    r.CMoveFor([0,0,0.1], 7)
    time.sleep(0.5)
    r.CMoveFor([0,0,0.4], 12)
    g.open()


###############################################################################################
#                   PICTURE TO WORLD COORDINATES CONVERSION
###############################################################################################

def pixel_to_world_coords(w, h, fx, fy, cx, cy, distance):
    """
    Convert pixel coordinates to real-world coordinates using the pinhole camera model.

    Parameters:
    - w, h: Pixel coordinates (width and height).c
    - fx, fy: Focal lengths in pixel units (derived from actual focal length and sensor size).
    - cx, cy: Principal point (usually the center of the image).
    - distance: Distance from the camera to the object in the Z-direction (depth).

    Returns:
    - X, Y, Z: Real-world coordinates.
    """
    X = (w - cx) * distance/fx
    Y = (h - cy) * distance/fy
    Z = distance
    return X, Y, Z


def read_csv(file_path):
    """
    Reads a CSV file and returns the pixel coordinates as a list of tuples with integers.

    Parameters:
    - file_path: Path to the CSV file.

    Returns:
    - corners: A list of tuples, each containing integer (w, h) values.
    """
    corners = []
    
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Convert each value in the row to an integer and append as a tuple
            corners.append((int(row[0]), int(row[1])))
    
    return corners

def get_world_coords():
    # Example usage
    file_path = 'docker/competition_output/17_corners.csv'  # Replace with your CSV file path
    corners = read_csv(file_path)

    # Print the list of tuples with the (w, h) values
    for i, (w, h) in enumerate(corners, start=1):
        print(f"Corner {i}: w = {w}, h = {h}")

    # Camera parameters
    image_width = 1920
    image_height = 1080
    focal_length_mm = 2.3  # Kinect depth camera focal length in mm // 1.8
    sensor_width_mm = 5.120  # Example sensor width in mm // 3.5
    sensor_height_mm = 3.132  # Example sensor height in mm // 2.0

    # Convert focal length to pixels
    fx = (focal_length_mm / sensor_width_mm) * image_width
    fy = (focal_length_mm / sensor_height_mm) * image_height

    # Assume the principal point is at the center of the image
    cx = image_width / 2
    cy = image_height / 2

    # Distance from camera to cloth in mm
    distance_mm = 1150  # 115 cm

    # Example corners of the cloth in pixel coordinates (w1, h1), (w2, h2), etc.
    #corners_pixels = [
    #    (w3, h3), (w2, h2), (w1, h1), (w4, h4)
    #]

    # Calculate real-world coordinates
    corners_world = []
    for (w, h) in corners:
        X, Y, Z = pixel_to_world_coords(w, h, fx, fy, cx, cy, distance_mm)
        corners_world.append((X, Y, Z))

    for i in corners_world:
        print("Real-world coordinates:", i) 
        #print("x: ", i[0], "y: ", i[1], "z: ", i[2])

    return corners_world

######################################################################################

if __name__ == "__main__":
        main()