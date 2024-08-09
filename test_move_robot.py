import sys
import time
#from icecream import ic
import numpy as np
sys.path.insert(1, '/home/peter/robotblockset_python')

from robotblockset_python.panda_ros import panda_ros
r = panda_ros('pingvin', init_node=True, init_frankadesk_gripper_TCP=True, start_controller='position_joint_trajectory_controller')


np.set_printoptions(formatter={"float": "{: 0.4f}".format})

while(True):
    # ic(r.F)
    r.GetState()
    force = np.array(r.FT)
    #ic(force)
    time.sleep(.1)

# p1.Switch_controller(start_controller='JointImpedance')
# print("going home position control")
# p1.JMove(p1.q_home, 5)
# # p1.Switch_controller(start_controller='JointImpedance')
# print("going down 10cm position control")

# p1.CMoveFor([0, 0, -0.1],5)
# print(p1.GetJointPos())
# print("switching to joint impedance")
# p1.Switch_controller(start_controller='JointImpedance')
# time.sleep(2)
# print("going up 20cm using impedance")
# p1.CMoveFor([0, 0, 0.2],5)
# print("going home using impedance")
# p1.JMove(p1.q_home, 5)
# print("finished")
