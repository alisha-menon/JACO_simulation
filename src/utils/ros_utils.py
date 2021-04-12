import rospy
import kinova_msgs.msg
import geometry_msgs.msg
from kinova_msgs.srv import *

def cmd_to_JointTorqueMsg(cmd):
	"""
	Returns a JointTorque Kinova msg from an array of torques
	"""
	jointCmd = kinova_msgs.msg.JointTorque()
	jointCmd.joint1 = cmd[0][0];
	jointCmd.joint2 = cmd[1][1];
	jointCmd.joint3 = cmd[2][2];
	jointCmd.joint4 = cmd[3][3];
	jointCmd.joint5 = cmd[4][4];
	jointCmd.joint6 = cmd[5][5];
	jointCmd.joint7 = cmd[6][6];

	return jointCmd

def cmd_to_JointAnglesMsg(cmd):
	"""
	Returns a JointVelocity Kinova msg from an array of velocities
	"""
	jointCmd = kinova_msgs.msg.JointAngles()
	jointCmd.joint1 = cmd[0][0];
	jointCmd.joint2 = cmd[1][1];
	jointCmd.joint3 = cmd[2][2];
	jointCmd.joint4 = cmd[3][3];
	jointCmd.joint5 = cmd[4][4];
	jointCmd.joint6 = cmd[5][5];
	jointCmd.joint7 = cmd[6][6];

	return jointCmd

def cmd_to_JointVelocityMsg(cmd):
	"""
	Returns a JointVelocity Kinova msg from an array of velocities
	"""
	jointCmd = kinova_msgs.msg.JointVelocity()
	jointCmd.joint1 = cmd[0][0];
	jointCmd.joint2 = cmd[1][1];
	jointCmd.joint3 = cmd[2][2];
	jointCmd.joint4 = cmd[3][3];
	jointCmd.joint5 = cmd[4][4];
	jointCmd.joint6 = cmd[5][5];
	jointCmd.joint7 = cmd[6][6];

	return jointCmd

def waypts_to_PoseArrayMsg(cart_waypts):
	"""
	Returns a PoseArray msg from an array of 3D carteian waypoints
	"""
	poseArray = geometry_msgs.msg.PoseArray()
	poseArray.header.stamp = rospy.Time.now()
	poseArray.header.frame_id = "/root"

	for i in range(len(cart_waypts)):
		somePose = geometry_msgs.msg.Pose()
		somePose.position.x = cart_waypts[i][0]
		somePose.position.y = cart_waypts[i][1]
		somePose.position.z = cart_waypts[i][2]

		somePose.orientation.x = 0.0
		somePose.orientation.y = 0.0
		somePose.orientation.z = 0.0
		somePose.orientation.w = 1.0
		poseArray.poses.append(somePose)

	return poseArray

def start_admittance_mode(prefix):
	"""
	Switches Kinova arm to admittance-control mode using ROS services.
	"""
	service_address = prefix+'/in/start_force_control'
	rospy.wait_for_service(service_address)
	try:
		startForceControl = rospy.ServiceProxy(service_address, Start)
		startForceControl()
	except rospy.ServiceException, e:
		print "Service call failed: %s"%e
		return None

def stop_admittance_mode(prefix):
    """
    Switches Kinova arm to position-control mode using ROS services.
    """
    service_address = prefix+'/in/stop_force_control'
    rospy.wait_for_service(service_address)
    try:
        stopForceControl = rospy.ServiceProxy(service_address, Stop)
        stopForceControl()
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
        return None
