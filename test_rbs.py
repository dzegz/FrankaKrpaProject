import rospkg
import rospy
from franka_msgs.msg import *

def callback(msg):
    print(msg)

rospy.init_node("test")
sub = rospy.Subscriber("/pingvin/franka_state_controller/franka_states", FrankaState, callback)
rospy.spin()