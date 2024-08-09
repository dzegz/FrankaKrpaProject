import numpy as np
#from icecream import ic
import time
import sys

sys.path.insert(1, '/home/peter/robotblockset_python')
from robotblockset_python.panda_ros import panda_ros
from robotblockset_python.ros.grippers_ros import PandaGripper
from robotblockset_python.transformations import rot_z
from robotblockset_python.paths import trajectories


r = panda_ros('pingvin', init_node=True, init_frankadesk_gripper_TCP=True, start_controller='position_joint_trajectory_controller')
time.sleep(1)
g = PandaGripper(robot=r, namespace='pingvin')
#ic("Gripper initialised")

# Rotate the frame of th TCP around axis z
# mat_rot_z = np.identity(4)
# mat_rot_z[0:3, 0:3] = rot_z(np.deg2rad(90))
# r.SetTCP(mat_rot_z @ r.TCP)

# predefined joint positions
q_start = np.array([
    -0.14046038953089038,
    0.6907892358709694,
    0.17331089070736802,
    -1.8385017293231556,
    0.8828741306767183,
    1.2778212858770368,
    -1.25
])

q_prestretch = np.array([
    1.5635650054367367,
    -0.977345715278542,
    -0.02086508573663869,
    -2.0256175952762896,
    0.005581337482498288,
    2.644842932851432,
    0.7993718486122142
])

# Predefined frame
T_prestretch = np.array([
    [ 0.99922433, -0.03567319,  0.01667837,  0.01564485],
    [-0.01574448,  0.02630163,  0.99953006,  0.28363888],
    [-0.0360951 , -0.99901734,  0.02571958,  0.90880785],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])

# Move in joint space TO specified joint position within 10 s
#ic("Going to pregrasp")
r.JMove(q_start, 10)

# Gripper homing action
g.homing()

# Set gripper width to 4 cm
g.move(0.04)

pose = np.array([
    [ 0.97199178,  0.12902535, -0.19642922,  0.06418075],
    [ 0.23329223, -0.63073377,  0.74010111,  0.31493582],
    [-0.02840274, -0.7651976 , -0.64316865,  0.47720053],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])

pose_stop = np.array([
    [ 0.99753617, -0.02704638, -0.06473089, -0.04647169],
    [ 0.06429051, -0.01684736,  0.997789  ,  0.09757993],
    [-0.02807712, -0.9994922 , -0.01506702,  0.90880785],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])

grasp_frame_approach_offset = -0.1
grasp_frame_approach_fine = -0.05
T_toolz_coarse = np.identity(4)
T_tool_z_fine = np.identity(4)
T_toolz_coarse[2, 3] = grasp_frame_approach_offset
T_tool_z_fine[2, 3] = grasp_frame_approach_fine

# Move in cartesian space TO POSITION in 15 s
#ic("Going to coarse pose")
r.CMove(pose @ T_toolz_coarse, 15)

#ic("Going to fine pose")
r.CMove(pose @ T_tool_z_fine, 15)

#ic("Going to pose")
r.CMove(pose, 15)
# g.grasp(0, speed=0.05, force=1)

# r.GoTo_TC(x=T_prestretch,
#           v=None,
#           FT=None,
#           null_space_task='ConfOptimization',
#           q_opt=q_prestretch)


r.CMove(T_prestretch, 10)
r.JMove(q_prestretch, 5)
r.CMove(pose_stop, 5)



# g = PandaGripper(robot=p1, namespace='pingvin')
# ic("Gripper initialised")
# ic(g)
# time.sleep(1)

# ic("homing")
# g.homing()

# width = 0.022
# ic(f"Move with width {width}")
# g.move(width)

# ic(f"Grasp with width {width}")
# g.grasp(width)

# ic("Closing")
# g.close()

# ic("Opening")
# g.open()


# ic("Homing position")
# p1.JMove(p1.q_home, 10)
# ic("going down 10cm position control")

# # ic("Joint position")
# # ic(p1.GetJointPos())
# # ic("--------------")
# # ic("TCP pose")
# # ic(p1.T)
# # ic("--------------")

# p1.CMove([0.5, 0, 0.5], 5)
# p1.CMove([0.4, 0, 0.6], 5)
# p1.CMoveFor([0, 0, -0.1], 5)
# p1.CMoveFor([0, 0, 0.2], 5)


# ic("switching to cartesian impedance")
# p1.Switch_controller(start_controller='CartesianImpedance')
# time.sleep(2)
# # p1.Start()
# # ic("switching to joint impedance")
# # p1.Switch_controller(start_controller='JointImpedance')
# # time.sleep(2)

# ic("set soft")
# p1.GetState()
# old_q = p1.q

# # p1.SetJointStiffness([100, 100, 100, 100, 100, 100, 100])

# p1.Stop()

# # p1.Soft()


# # p1.GetState()
# # p1.SetJointDamping([100, 100, 100, 100, 100, 100, 100])

# ic("sleeping for 15 s")
# time.sleep(25)
# p1.GetState()
# new_q = p1.q

# ic("old q")
# ic(old_q)
# ic("new q")
# ic(new_q)


# # p1.Stiff()

# while True:
#     break
#     p1.GetState()

#     p1._command_int.q = p1._actual_int.q
#     p1._command_int.qdot = p1._actual_int.qdot  # *0
#     p1._command_int.trq = p1._actual_int.trq   # *0
#     p1._command_int.p = p1._actual_int.p
#     p1._command_int.R = p1._actual_int.R
#     p1._command_int.x = p1._actual_int.x
#     p1._command_int.v = p1._actual_int.v    # *0

#     cmd_msg = p1.empty_JointCommand
#     cmd_msg.pos = p1._actual_int.q
#     cmd_msg.vel = p1._actual_int.qdot
#     cmd_msg.trq = p1._actual_int.trq
#     cmd_msg.impedance.n = 7
#     cmd_msg.impedance.k = p1.joint_compliance.K
#     cmd_msg.impedance.d = p1.joint_compliance.D

#     p1.joint_command_publisher.publish(cmd_msg)
#     p1.Update()
#     # break


# p1.Stop()
#ic("finished")
