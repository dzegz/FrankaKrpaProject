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

# pddl

import unified_planning as up
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import *

import re


###############################################################################################
#                           PROCEDURE SCRIPTS
###############################################################################################

def grip_handler(r, g, current_point, parsed_action):
    # position before movement
    #translation_target0, orientation_target0 = self._target.get_world_pose()
    
    # orientation target calculation
    # P2 = self._close_points[1]
    # P3 = self._close_points[0]
    # theta = math.pi - math.atan2(P3[1]-current_point[1],P3[0]-current_point[0])-math.atan2(P2[1]-current_point[1],P2[0]-current_point[0])
    # rotation_quaternion = R.from_euler('x', theta).as_quat()
    # new_orient = self.quaternion_multiply(orientation_target0, rotation_quaternion)
    
    # if want_orientation:
    #     orientation_target = new_orient
    #     print("Orientation target changed | New OT: ", new_orient)
    
    #self._target.set_world_pose(current_point, orientation_target)

    #success = yield from self.goto_position(
    #    current_point, orientation_target, self._articulation, self._rmpflow, timeout=250
    #)

    #yield from self.close_gripper_franka(self._articulation)
    print(f"Going to {parsed_action['parameters'][1]} - {current_point}")
    r.CMove([current_point[0], current_point[1], 0.05], 15)
    #time.sleep(1)
    r.CMove([current_point[0], current_point[1], 0.02], 5)
    #time.sleep(1)
    g.close()

def p2p_handler(r, g, current_point, target_point_id, parsed_action):
    print(f"Going from {parsed_action['parameters'][1]} to {parsed_action['parameters'][2]} - {current_point}")
    
    target_points = get_world_coords()
    # for better path
    middle_point = (current_point+target_points[target_point_id])/2
    middle_point[2] = 0.15

    print("Going up middle: ", middle_point)
    #self._target.set_world_pose(middle_point, orientation_target)
    r.CMove(middle_point, 10)

    # success = yield from self.goto_position(
    #     middle_point, orientation_target, self._articulation, self._rmpflow, timeout=250
    # )

    # final target (next point pX)
    #yield from self.particle_script()
    target_points = get_world_coords()

    final_point = target_points[target_point_id]
    print("Going to destination", final_point)
    #self._target.set_world_pose(final_point, orientation_target)
    r.CMove([final_point[0], final_point[1], 0.05], 15)
    r.CMove([final_point[0], final_point[1], 0.03], 5)
    #time.sleep(1)

    # success = yield from self.goto_position(
    #     final_point, orientation_target, self._articulation, self._rmpflow, timeout=2500     
    # )

def understanding_script(r, g, result):
    plan_str = str(result.plan)
    actions = plan_str.split('\n')  
    parsed_actions = []
    action_pattern = re.compile(r'(\w+)\(([^)]+)\)')

    # Parsing each action
    for action in actions:
        match = action_pattern.match(action.strip())
        if match:
            action_name = match.group(1)
            parameters = [param.strip() for param in match.group(2).split(',')]
            parsed_actions.append({'action': action_name, 'parameters': parameters})

    # Printing the parsed actions
    #for parsed_action in parsed_actions:
    #    print(f"Action: {parsed_action['action']}, Parameters: {parsed_action['parameters']}")
    

    # handling the acquired actions
    #current_point, orientation_target = self._target.get_world_pose()
    
    for parsed_action in parsed_actions:
        #refreshing the positions
        #time.sleep(0.5)
        target_points = get_world_coords()


        action_name = parsed_action['action']
        parameters = parsed_action['parameters']
        if action_name == "grip":
            if parameters == ['g', 'p1']:
                print("Grip action with parameters g, p1")
                current_point = target_points[0]
                grip_handler(r, g, current_point, parsed_action)
                
            elif parameters == ['g', 'p3']:
                print("Grip action with parameters g, p3")
                current_point = target_points[2]
                grip_handler(r, g, current_point, parsed_action)

        elif action_name == "p2p":
            if parameters == ['g', 'p1', 'p2']:
                print("p2p action with parameters g, p1, p2")
                current_point = target_points[0]
                p2p_handler(r, g, current_point, 1, parsed_action)
            elif parameters == ['g', 'p3', 'p4']:
                print("p2p action with parameters g, p3, p4")
                current_point = target_points[2]
                p2p_handler(r, g, current_point, 3, parsed_action)

        elif action_name == "release":
            if parameters == ['g', 'p1']:
                print("Release action with parameters g, p1")
                #yield from self.open_gripper_franka(self._articulation)
                g.open()
                #time.sleep(0.5)
            elif parameters == ['g', 'p3']:
                print("Release action with parameters g, p3")
                g.open()
                #yield from self.open_gripper_franka(self._articulation)
                #time.sleep(0.5)

            # go up after releasing
            r.CMoveFor([0, 0, 0.15], 7)
            #translation_target, orientation_target = self._target.get_world_pose()
            #self._target.set_world_pose(translation_target+np.array([0, 0, 0.1]), orientation_target)

            # success = yield from self.goto_position(
            #     translation_target+np.array([0, 0, 0.1]), orientation_target, self._articulation, self._rmpflow, timeout=2500     
            # ) 
        else:
            print(f"Unknown action: {action_name} with parameters {parameters}")
    

def planning_script():
    domain_file = "domain.pddl"
    problem_file = "problem.pddl"

    reader = PDDLReader()
    pddl_problem = reader.parse_problem(domain_file, problem_file)

    # disabling the credits
    up.shortcuts.get_environment().credits_stream = None

    # solving the problem
    with OneshotPlanner(problem_kind=pddl_problem.kind) as planner:
        result = planner.solve(pddl_problem)
        print("%s returned: %s" % (planner.name, result.plan))

    return result


def main():
    #corners = get_world_coords()
    
    ####

    r = panda_ros('pingvin', init_node=True, init_frankadesk_gripper_TCP=True, start_controller='position_joint_trajectory_controller')
    time.sleep(1)
    g = PandaGripper(robot=r, namespace='pingvin')

    # going to the initial position
    q_home= np.array([0, -0.2, 0, -1.5, 0, 1.5, 0.7854]) # home joint configuration
    r.JMove(q_home, 10)

    robot_points = get_world_coords()
    result = planning_script()

    g.homing()
    g.open()
    understanding_script(r, g, result)

    # # go out of the camera frame
    # pose = np.array([
    #     [ 0.97199178,  0.12902535, -0.19642922,  0.06418075],
    #     [ 0.23329223, -0.63073377,  0.74010111,  0.31493582],
    #     [-0.02840274, -0.7651976 , -0.64316865,  0.47720053],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]
    # ])
    # r.CMove(pose, 10)

    # # at this moment you should take a picture and run the pddl planning procedure
    # time.sleep(3)

    # # going back to the initial position
    # r.JMove(q_home, 10)
    # #going to the center of the coordinate system
    # r.CMove([0.505,0.115,0.4], 10) # center of the camera system

    # # gripper homing
    # g.homing()
    # g.open()

    # # going down
    # r.CMoveFor([0,0,-0.6], 10)
    # time.sleep(1)
    # r.CMoveFor([0,0,-0.025], 3)
    # time.sleep(1)

    # # grabbing
    # g.close()

    # # going up and releasing
    # r.CMoveFor([0,0,0.1], 7)
    # time.sleep(0.5)
    # r.CMoveFor([0,0,0.4], 12)
    # g.open()


###############################################################################################
#                   PICTURE TO WORLD COORDINATES CONVERSION
###############################################################################################

def transform_points_to_robot_coords(points):
    # Convert list of tuples to a numpy array for easier manipulation
    points = np.array(points)
    
    # Apply the transformation to each point
    transformed_points = np.array([(y, x, z) for x, y, z in points])
    transformed_points = transformed_points + np.array([0.505, 0.115, 0])

    return transformed_points

def sort_cloth_corners(points):
    # Convert list of tuples to a numpy array for easier manipulation
    points = np.array(points)
    
    # Sort by y-coordinate to separate upper and lower points
    sorted_by_y = points[points[:, 1].argsort()]
    
    # Assume the first two points are upper, last two are lower
    upper_points = sorted_by_y[:2]
    lower_points = sorted_by_y[2:]
    
    # Sort upper points by x-coordinate to get left and right
    upper_left, upper_right = upper_points[upper_points[:, 0].argsort()]
    
    # Sort lower points by x-coordinate to get left and right
    lower_left, lower_right = lower_points[lower_points[:, 0].argsort()]
    
    # Return sorted points in the order: upper left, lower left, upper right, lower right
    return np.array([upper_left, lower_left, upper_right, lower_right])

def pixel_to_world_coords(w, h, fx, fy, cx, cy, distance):
    """
    Convert pixel coordinates to real-world coordinates using the pinhole camera model.

    Parameters:
    - w, h: Pixel coordinates (width and height).
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
    file_path = 'docker/competition_output/24_corners.csv'  # Replace with your CSV file path
    corners = read_csv(file_path)

    # Print the list of tuples with the (w, h) values
    for i, (w, h) in enumerate(corners, start=1):
        print(f"Corner {i}: w = {w}, h = {h}")

    # Camera parameters
    image_width = 1920
    image_height = 1080
    focal_length_mm = 2.3 # Kinect depth camera focal length in mm // 1.8
    sensor_width_mm = 5.120  # Example sensor width in mm // 3.5
    sensor_height_mm = 3.132 # Example sensor height in mm // 2.0

    # Convert focal length to pixels
    #fx = (focal_length_mm / sensor_width_mm) * image_width
    #fy = (focal_length_mm / sensor_height_mm) * image_height
    FOV_h = 90 #75 65
    FOV_v = 59
    fx = (image_width * 0.5) / np.tan(FOV_h * 0.5 * np.pi/180)
    fy = (image_height * 0.5) / np.tan(FOV_v * 0.5 * np.pi/180)

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

    # for i in corners_world:
    #     print("Real-world coordinates:", i) s
    #     #print("x: ", i[0], "y: ", i[1], "z: ", i[2])

    print("p1: ", corners_world[2])
    print("p2: ", corners_world[1])
    print("p3: ", corners_world[0])
    print("p4: ", corners_world[3])
    print("----------------------------------- sorted points")
    sorted_points = sort_cloth_corners(corners_world)/1000
    print("p1: ", sorted_points[0])
    print("p2: ", sorted_points[1])
    print("p3: ", sorted_points[2])
    print("p4: ", sorted_points[3])
    print("----------------------------------- robot coords")
    robot_points = transform_points_to_robot_coords(sorted_points)   

    print("p1: ", robot_points[0])
    print("p2: ", robot_points[1])
    print("p3: ", robot_points[2])
    print("p4: ", robot_points[3])

    return robot_points

######################################################################################

if __name__ == "__main__":
    main()

#planning_script()