#! /usr/bin/env python
"""


Authors: Andreea Bobu (abobu@eecs.berkeley.edu), Andrea Bajcsy (abajcsy@eecs.berkeley.edu), Matthew Zurek
"""

import math
import sys, select, os
import time
from threading import Thread

#from utils.environment_utils import *

import pybullet as p
import numpy as np
import torch
from controllers.pid_controller import PIDController
from learners.teleop_learner import TeleopLearner
from utils.environment_utils import *


from teleop_inference_base import TeleopInferenceBase


CONFIG_FILE_DICT = {
	0: {
		'demos': '../config/task0_methodd_inference_config.yaml',
		'ex': "config/task0_example_inference_config.yaml",
		'a': "config/task0_methoda_inference_config.yaml",
		'b': "config/task0_methodb_inference_config.yaml",
		'c': "config/task0_methodc_inference_config.yaml"},
	1: {
		'ex': "config/task1_example_inference_config.yaml",
		'a': "config/task1_methoda_inference_config.yaml",
		'b': "config/task1_methodb_inference_config.yaml",
		'c': "config/task1_methodc_inference_config.yaml"
	},
	2: {
		'ex': "config/task2_example_inference_config.yaml",
		'a': "config/task2_methoda_inference_config.yaml",
		'b': "config/task2_methodb_inference_config.yaml",
		'c': "config/task2_methodc_inference_config.yaml",
		'demos': "config/task2_methodd_inference_config.yaml",
		'c_learned': "config/task2_methode_inference_config.yaml"
	},
	-100: { # don't use this one
		'a': "config/task3_methoda_inference_config.yaml",
		'b': "config/task3_methodb_inference_config.yaml",
		'c': "config/task3_methodc_inference_config.yaml",
		'd': "config/task3_methodd_inference_config.yaml",
		'e': "config/task3_methode_inference_config.yaml"
	},
	3: {
		'ex': "config/task4_example_inference_config.yaml",
		'a': "config/task4_methoda_inference_config.yaml",
		'b': "config/task4_methodb_inference_config.yaml",
		'c': "config/task4_methodc_inference_config.yaml",
		'demos': "config/task4_methodd_inference_config.yaml",
		'c_learned': "config/task4_methode_inference_config.yaml"
	}
}

def topview1():
	p.resetDebugVisualizerCamera(cameraDistance=1.50, cameraYaw=90, cameraPitch=-70, cameraTargetPosition=[-0.4, -0.25, -0.05])

def topview2():
	p.resetDebugVisualizerCamera(cameraDistance=1.50, cameraYaw=90, cameraPitch=-89, cameraTargetPosition=[-0.7, -0.25, -0.05])

def sideview():
	p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[-0.6, 1, -0.05])


class TeleopInference(TeleopInferenceBase):
	"""
	This class represents a node that moves the Jaco with PID control AND supports receiving human corrections online.
	"""

	def __init__(self, config_file):

		self.state_based = 1

		super(TeleopInference, self).__init__(False, config_file)
		config = self.config

		# ------- setup pybullet -------
		physicsClient = p.connect(p.GUI)
		#physicsClient = p.connect(p.GUI, options="--opengl2")


		p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, lightPosition=[-0.4, -0.25,10])

		# Add path to data resources for the environment.
		p.setAdditionalSearchPath("../data/resources")

		# Setup the environment.
		self.bullet_environment = setup_environment(self.visual_goals)
		top1Button = p.addUserDebugParameter("Top-down View #1", 1, 0, 0)
		top2Button = p.addUserDebugParameter("Top-down View #2", 1, 0, 0)
		sideButton = p.addUserDebugParameter("Side View", 1, 0, 0)
		#self.bullet_environment = setup_environment(self.goals)

		# load learned goals
		for learned_goal_save_path in config["setup"]["learned_goals"]:
			self.goals.append(torch.load(learned_goal_save_path)['goal'])

		# Calculate goal locations in xyz
		self.goal_locs = []
		for goal in self.goals:
			move_robot(self.bullet_environment["robot"], np.append(goal.reshape(7), np.array([0, 0, 0])))
			self.goal_locs.append(robot_coords(self.bullet_environment["robot"])[-1])

		# update IK of goals
		bullet_start = np.append(self.start.reshape(7), np.array([0.0, 0.0, 0.0]))
		move_robot(self.bullet_environment["robot"], bullet_start)
		self.update_IK_goals()

		# Get rid of gravity and make simulation happen in real time.
		p.setGravity(0, 0, -10)
		p.setRealTimeSimulation(1)

		# ----- Learner Setup ----- #
		betas = np.array(config["learner"]["betas"])
		goal_beliefs = config["learner"]["goal_beliefs"]
		if goal_beliefs != "none":
			goal_beliefs = goal_beliefs / np.linalg.norm(goal_beliefs)
		else:
			goal_beliefs = np.ones(self.num_goals)/self.num_goals
		assert(len(goal_beliefs) == self.num_goals)
		beta_priors = config["learner"]["beta_priors"]
		if beta_priors == "none":
			beta_priors = np.zeros(self.num_goals)
		assert(len(goal_beliefs) == self.num_goals)
		self.inference_method = config["learner"]["inference_method"]
		self.beta_method = config["learner"]["beta_method"]
		self.learner = TeleopLearner(self, goal_beliefs, beta_priors, betas, self.inference_method, self.beta_method)
		self.running_inference = False
		self.last_inf_idx = 0
		self.running_final_inference = False
		self.final_inference_done = False

		self.exp_data['inf_start_time'] = []
		self.exp_data['joy_cmd'] = []
		self.exp_data['ctl_cmd'] = []
		self.exp_data['cmd'] = []
		self.exp_data['cmdpos_time'] = []
		self.exp_data['curr_pos'] = []
		self.exp_data['num_key_presses'] = []


		print ("----------------------------------")
		print("Simulating robot, press ENTER to quit:")
		# Start simulation.
		if self.inference_method == "collect":
			N = 5

			# Add demonstration recording buttons.
			self.buttons = [p.addUserDebugParameter("Stop Recording", 1, 0, 0),
							p.addUserDebugParameter("Next Demo", 1, 0, 0),
							p.addUserDebugParameter("Save Demo", 1, 0, 0)]
			self.numPush = [0, 0, 0]
		else:
			N = 1
		self.queries = 0
		self.recorded_demos = []

		sideview()
		top1Num = 0
		top2Num = 0
		sideNum = 0

		#state machine
		# 1 = aligniing EE
		# 2 = opening fingers + height
		# 3 = making contact
		# 4 = gripping object
		# 5 = lifting object

		current_state = 1
		next_state = 1

		while self.queries < N:
			print ("Attempting round {}.".format(self.queries+1))
			move_robot(self.bullet_environment["robot"], bullet_start)
			self.demo = [np.append(np.array([0.0]), bullet_start)]
			self.running = True
			self.record = True

			start_time = time.time()
			self.num_key_presses = 0
			# Start simulation.
			while self.running:
				current_state = next_state
				top1Pushes = p.readUserDebugParameter(top1Button)
				top2Pushes = p.readUserDebugParameter(top2Button)
				sidePushes = p.readUserDebugParameter(sideButton)
				if top1Pushes > top1Num:
					print("Changing the view to top-down #1.")
					top1Num = top1Pushes
					topview1()

				if top2Pushes > top2Num:
					print("Changing the view to top-down #2.")
					top2Num = top2Pushes
					topview2()

				if sidePushes > sideNum:
					print("Changing the view to side.")
					sideNum = sidePushes
					sideview()

				if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
					line = raw_input()
					break

				font_size = 0 if self.inference_method == "collect" else 2
				time_now = time.time() - start_time
				if 'time_text' in locals():
					p.removeUserDebugItem(time_text)
				time_text = p.addUserDebugText("{:.2f}s".format(np.clip(15-time_now,0,15)), [-1,0.75,1],textSize=font_size)
				if 'keys_text' in locals():
					p.removeUserDebugItem(keys_text)
				keys_text = p.addUserDebugText("{} keys".format(self.num_key_presses),[-1,0.75,0.75],textSize=font_size)

				# Update position.
				if self.state_based == 1:
					if current_state == 1:
						next_state = self.keyboard_input_callback_state1()
					elif current_state == 2:
						next_state = self.keyboard_input_callback_state2()
					elif current_state == 3:
						next_state = self.keyboard_input_callback_state3()
					elif current_state == 4:
						next_state = self.keyboard_input_callback_state4()
					elif current_state == 5:
						next_state = self.keyboard_input_callback_state5()
					else:
						next_state = self.keyboard_input_callback_state1()
				else:
					self.keyboard_input_callback()
				#print(current_state)

				# Look after button presses.
				if self.inference_method == "collect":
					self.update_buttons()
				else:
					self.queries = 1

				time.sleep(0.05)


		if self.inference_method == "collect":
			# Save FK and IK.
			FKs = []
			for demo in self.recorded_demos:
				goal = np.append(demo[-1].reshape(7), np.array([0.0, 0.0, 0.0]))
				move_robot(self.bullet_environment["robot"], goal)
				FKs.append(robot_coords(self.bullet_environment["robot"])[-1])
			avg_goal = np.mean(np.array(FKs), axis=0)
			avg_angles = p.calculateInverseKinematics(self.bullet_environment["robot"], 7, avg_goal)
			np.savez(config['setup']['demonstrations_save_path'], demos=self.recorded_demos, FK_goal=avg_goal, IK_goal=avg_angles[:7], FKs=FKs)

		# Disconnect once the session is over.
		p.disconnect()
		if self.inference_method != "collect":
			self.exp_data['traj_hist'] = self.traj_hist
			self.exp_data['start_T'] = self.start_T
			np.savez(config['setup']['data_save_path'],
					 beta_hist=self.exp_data['beta_hist'],
					 sub_hist=self.exp_data['sub_hist'],
					 belief_hist=self.exp_data['belief_hist'],
					 goal_costs=self.exp_data['goal_costs'],
					 curr_cost=self.exp_data['curr_cost'],
					 goal_traj_by_idx=self.exp_data['goal_traj_by_idx'],
					 goal_traj_plan_by_idx=self.exp_data['goal_traj_plan_by_idx'],
					 traj_hist=self.exp_data['traj_hist'],
					 start_T=self.exp_data['start_T'],
					 inf_start_time=self.exp_data['inf_start_time'],
					 joy_cmd=self.exp_data['joy_cmd'],
					 ctl_cmd=self.exp_data['ctl_cmd'],
					 cmd=self.exp_data['cmd'],
					 cmdpos_time=self.exp_data['cmdpos_time'],
					 curr_pos=self.exp_data['curr_pos'],
					 num_key_presses=self.exp_data['num_key_presses'],
					 optimal_costs=self.exp_data['optimal_costs'])


		print ("----------------------------------")

	def joint_angles_callback(self, curr_pos):
		"""
		Reads the latest position of the robot and publishes an
		appropriate velocity command to move the robot to the target.
		"""
		# Convert to radians.
		self.curr_pos = curr_pos*(math.pi/180.0)
		#print 'curr_pos', curr_pos

		if self.start_T is not None and (time.time() - self.start_T >= self.timestep * self.next_waypt_idx):
			if not self.next_waypt_idx >= len(self.traj_hist):
				self.traj_hist[self.next_waypt_idx] = self.curr_pos.reshape(7)
				self.next_waypt_idx += 1
				#print "timestep:", self.next_waypt_idx
				if not self.running_inference and self.next_waypt_idx - 1 != len(self.traj_hist) - 1: # second condition: don't call inference with 1 timestep left
					#print 'calling inference from', self.next_waypt_idx - 1

					self.update_IK_goals()
					#print 'original goal distance:', np.linalg.norm(self.goals - self.curr_pos, axis=1)
					#print 'IK goal distance:', np.linalg.norm(self.IK_goals - self.curr_pos, axis=1)

					self.running_inference = True
					self.inference_thread = Thread(target=self.learner.inference_step)
					self.inference_thread.start()
					self.exp_data['inf_start_time'].append(time.time() - self.start_T)
			elif not self.running_final_inference:
				self.running_final_inference = True
				self.inference_thread.join()
				self.inference_thread = Thread(target=self.learner.final_step)
				self.inference_thread.start()

				print ("Episode over. Running final calculations.")
				self.inference_thread.join()


				self.running = False

			elif self.final_inference_done:
				pass

		ctl_cmd = self.controller.get_command(self.curr_pos.reshape(7,1))
		self.exp_data['joy_cmd'].append(np.copy(self.joy_cmd[np.arange(7), np.arange(7)]))
		self.exp_data['cmdpos_time'].append(time.time() - self.start_T)
		self.exp_data['curr_pos'].append(np.copy(self.curr_pos))
		self.exp_data['num_key_presses'].append(self.num_key_presses)

		if self.assistance_method == "blend":
			if self.learner.last_inf_idx > self.last_inf_idx: # new inference step complete
				self.last_inf_idx = self.learner.last_inf_idx
				if self.beta_method == "joint":
					goal, beta = self.learner.argmax_joint_beliefs
					#print 'goal:', goal, 'beta:', beta
					print ('joint beliefs:', self.learner.joint_beliefs)
					belief = np.max(self.learner.joint_beliefs[goal])
				elif self.beta_method == "estimate":
					goal, beta = self.learner.argmax_estimate
					belief = self.learner.goal_beliefs[goal]
					#print 'goal:', goal, 'beta:', beta
					#print 'beta estimates:', self.learner.beta_estimates
					#print 'goal beliefs:', self.learner.goal_beliefs
				self.alpha = self.beta_arbitration(beta, belief, goal)
				print ('alpha:', self.alpha)
				#if goal != self.curr_goal or self.alpha_method != "zero":
				if goal != self.curr_goal:
					print ('new assistance trajectory, goal:', goal)
					self.curr_goal = goal
					self.traj = self.learner.cache['goal_traj_by_idx'][self.last_inf_idx][goal]

					self.traj_plan = self.learner.cache['goal_traj_plan_by_idx'][self.last_inf_idx][goal]
					self.controller.set_trajectory(self.traj,
												   path_start_T=self.idx_to_time(self.last_inf_idx))
			if np.allclose(self.joy_cmd, np.zeros((7,7))) and not self.zero_input_assist:
				self.cmd = self.joy_cmd
			else:
				self.cmd = self.alpha * self.joy_cmd + (1. - self.alpha) * ctl_cmd

			# logging
			self.exp_data['ctl_cmd'].append(np.copy(ctl_cmd[np.arange(7), np.arange(7)]))
			self.exp_data['cmd'].append(np.copy(self.cmd[np.arange(7), np.arange(7)]))
		elif self.assistance_method == "expected":
			raise NotImplementedError
		elif self.assistance_method == "none":
			if self.learner.last_inf_idx > self.last_inf_idx: # new inference step complete
				self.last_inf_idx = self.learner.last_inf_idx
			self.cmd = self.joy_cmd
		else:
			raise ValueError
		# Update cmd from PID based on current position.
		#self.cmd = self.controller.get_command(self.curr_pos)

	def replay_trace(self, trace):
		for waypt in trace:
			for jointIndex in range(p.getNumJoints(self.bullet_environment["robot"])):
				p.resetJointState(self.bullet_environment["robot"], jointIndex, waypt[jointIndex])
			time.sleep(0.01)

	def update_buttons(self):
		stopPushes = p.readUserDebugParameter(self.buttons[0])
		nextPushes = p.readUserDebugParameter(self.buttons[1])
		savePushes = p.readUserDebugParameter(self.buttons[2])

		if stopPushes > self.numPush[0]:
			print ("Stopping recording. If happy with the recording, press Save Demo; otherwise press Next Demo.")
			self.numPush[0] = stopPushes
			if self.record == True:
				self.record = False
				# Pre-process the recorded data.
				trace = np.squeeze(np.array(self.demo))
				lo = 0
				hi = trace.shape[0] - 1
				while lo < hi and np.linalg.norm(trace[lo] - trace[lo + 1]) < 0.001:
					lo += 1
				while hi > 0 and np.linalg.norm(trace[hi] - trace[hi - 1]) < 0.001:
					hi -= 1
				trace = trace[lo:hi+1, :]
				self.replay_trace(trace)
				self.trace = trace
			else:
				print ("Can't stop a recording that hasn't started yet.")

		if nextPushes > self.numPush[1]:
			self.numPush[1] = nextPushes
			self.running = False

		if savePushes > self.numPush[2]:
			self.numPush[2] = savePushes
			if self.record == False:
				# Save trajectory.
				traj = np.array(self.trace)[:,1:8]
				self.recorded_demos.append(traj)
				print ('Saved trajectory {}.'.format(self.queries+1))
				self.queries += 1
				self.running = False
			else:
				print ("Can't save while recording! Please stop the recording first.")

		if self.record:
			state = p.getJointStates(self.bullet_environment["robot"],
									 range(p.getNumJoints(self.bullet_environment["robot"])))
			waypt = [s[0] for s in state]
			self.demo.append(np.array(waypt))


	def keyboard_input_callback_state1(self):
		# Reset variables.
		# setup y and z to be within object margin
		jointVelocities = [0.0] * p.getNumJoints(self.bullet_environment["robot"])
		dist_step = [0.0025, 0.0025, 0.0025]
		time_step = 0.05
		turn_step = 0.025
		EElink = 7
		next_state = 1
		y_margin = 0.1
		z_margin = 0.16

		# Get current EE position. 
		#robot_coords only gives the 7 degrees of freedom, EEpos is only the position of the end effector. 

		all_coords = robot_coords(self.bullet_environment["robot"])
		#print(all_coords) 
		#there are 11 with 3 coordinates each
		EEPos = robot_coords(self.bullet_environment["robot"])[EElink-1]
		self.EEPos = EEPos

		state = p.getJointStates(self.bullet_environment["robot"], range(p.getNumJoints(self.bullet_environment["robot"])))
		jointPoses = np.array([s[0] for s in state])

		# Parse keyboard commands.
		EEPos_new = np.copy(EEPos)
		keys = p.getKeyboardEvents()
		self.num_key_presses += len(keys)
		if p.B3G_LEFT_ARROW in keys:
			EEPos_new[1] -= dist_step[1]
		if p.B3G_RIGHT_ARROW in keys:
			EEPos_new[1] += dist_step[1]
		#if p.B3G_UP_ARROW in keys:
		#	EEPos_new[2] += dist_step[2]
		#if p.B3G_DOWN_ARROW in keys:
		#	EEPos_new[2] -= dist_step[2]


		# Get new velocity.
		if not np.array_equal(EEPos_new, EEPos):
			newPoses = np.asarray((0.0,) + p.calculateInverseKinematics(self.bullet_environment["robot"], EElink, EEPos_new))
			#print(len(newPoses))
			jointVelocities = (newPoses - jointPoses) / time_step

		# Update joystick command.
		self.joy_cmd = np.diag(jointVelocities[1:11])

		if not (self.inference_method == "collect"):
			# Move arm in openrave as well.
			joint_angles = jointPoses[1:8] * (180/np.pi)
			self.joint_angles_callback(joint_angles)
		else:
			# THIS IS THE ONE THAT IS HAPPENING
			self.cmd = self.joy_cmd

		# Update sim position with new velocity command.
		for i in range(len(self.cmd)):
			p.setJointMotorControl2(self.bullet_environment["robot"], i+1, p.VELOCITY_CONTROL, targetVelocity=self.cmd[i][i])

		EEPos = robot_coords(self.bullet_environment["robot"])[EElink-1]
		all_cup_coords = cup_coords(self.bullet_environment["mug"])	

		#print(EEPos[1],EEPos[2])	
		#print(all_cup_coords[1],all_cup_coords[2])
		#print distance
		#print(abs(EEPos[1] - all_cup_coords[1]),abs(EEPos[2] - all_cup_coords[2]))
		# if (abs(EEPos[1] - all_cup_coords[1]) < y_margin) and (abs(EEPos[2] - all_cup_coords[2]) < z_margin):
		# 	print("within y and z margin, switch to state 2")
		# 	next_state = 2
		# elif (abs(EEPos[1] - all_cup_coords[1]) < y_margin) and not (abs(EEPos[2] - all_cup_coords[2]) < z_margin):
		# 	print("within y margin")
		# elif (abs(EEPos[1] - all_cup_coords[1]) < y_margin) and not (abs(EEPos[2] - all_cup_coords[2]) < z_margin):
		# 	print("within z margin")

		if abs(EEPos[1] - all_cup_coords[1]) < y_margin:
			print("within y margin")
			next_state = 2


		return next_state

	def keyboard_input_callback_state2(self):
		# open hand wide enough for object
		# Reset variables.
		jointVelocities = [0.0] * p.getNumJoints(self.bullet_environment["robot"])
		dist_step = [0.0025, 0.0025, 0.0025]
		time_step = 0.05
		turn_step = 0.025
		EElink = 7
		next_state = 2
		distance_margin = 0.06
		finger_length = 0.06
		distance_between_fingers = 0.041

		# Get current EE position. 
		#robot_coords only gives the 7 degrees of freedom, EEpos is only the position of the end effector. 

		all_coords = robot_coords(self.bullet_environment["robot"])
		#print(all_coords) 
		#there are 11 with 3 coordinates each
		EEPos = robot_coords(self.bullet_environment["robot"])[EElink-1]
		self.EEPos = EEPos

		state = p.getJointStates(self.bullet_environment["robot"], range(p.getNumJoints(self.bullet_environment["robot"])))
		jointPoses = np.array([s[0] for s in state])

		# Parse keyboard commands.
		EEPos_new = np.copy(EEPos)
		keys = p.getKeyboardEvents()
		self.num_key_presses += len(keys)

		# Get new velocity.
		if not np.array_equal(EEPos_new, EEPos):
			newPoses = np.asarray((0.0,) + p.calculateInverseKinematics(self.bullet_environment["robot"], EElink, EEPos_new))
			#print(len(newPoses))
			jointVelocities = (newPoses - jointPoses) / time_step
		if p.B3G_LEFT_ARROW in keys:
			jointVelocities[EElink+2] += turn_step / time_step
			jointVelocities[EElink+3] += turn_step / time_step
			jointVelocities[EElink+1] += turn_step / time_step
		if p.B3G_RIGHT_ARROW in keys:
			jointVelocities[EElink+2] -= turn_step / time_step
			jointVelocities[EElink+3] -= turn_step / time_step
			jointVelocities[EElink+1] -= turn_step / time_step			
		# Update joystick command.
		self.joy_cmd = np.diag(jointVelocities[1:11])

		if not (self.inference_method == "collect"):
			# Move arm in openrave as well.
			joint_angles = jointPoses[1:8] * (180/np.pi)
			self.joint_angles_callback(joint_angles)
		else:
			# THIS IS THE ONE THAT IS HAPPENING
			self.cmd = self.joy_cmd

		# Update sim position with new velocity command.
		for i in range(len(self.cmd)):
			p.setJointMotorControl2(self.bullet_environment["robot"], i+1, p.VELOCITY_CONTROL, targetVelocity=self.cmd[i][i])

		#fingers have 114 degree range of motion, joint poses gives angle in radians (range of 2 radians), not seeing any change in position
		right_finger1 = all_coords[EElink+1]
		right_finger2 = all_coords[EElink+2]
		left_finger = all_coords[EElink]
		all_cup_coords = cup_coords(self.bullet_environment["mug"])

		#print(EEPos[1],EEPos[2])	
		#print(right_finger1)
		#print(right_finger2)
		#print(left_finger)
		#print(all_coords)
		state = p.getJointStates(self.bullet_environment["robot"], range(p.getNumJoints(self.bullet_environment["robot"])))
		jointPoses = np.array([s[0] for s in state])
		distance = math.sin(0.7 - jointPoses[EElink+1])*finger_length*2 + distance_between_fingers

		#print(jointPoses)
		#print(jointPoses[EElink+1])
		#print(jointPoses[EElink+2])
		#print(jointPoses[EElink+3])
		#print(all_cup_coords[1],all_cup_coords[2])
		#print distance

		# can we translate joint angle to a radius of grip which can be used?
		print(distance)

		#print(abs(right_finger1[1] - left_finger[1]),abs(right_finger2[1] - left_finger[1]))

		if (distance > distance_margin):
			print("wide enough grasp distance, switch to state 3")
			next_state = 3

		return next_state

	def keyboard_input_callback_state3(self):
		# Reset variables.
		jointVelocities = [0.0] * p.getNumJoints(self.bullet_environment["robot"])
		dist_step = [0.0025, 0.0025, 0.0025]
		time_step = 0.05
		turn_step = 0.02
		EElink = 7
		next_state = 3
		y_contact_threshold = 0.05
		x_contact_threshold = 0.145
		z_contact_threshold = 0.1

		# Get current EE position. 
		#robot_coords only gives the 7 degrees of freedom, EEpos is only the position of the end effector. 

		all_coords = robot_coords(self.bullet_environment["robot"])
		#print(all_coords) 
		#there are 11 with 3 coordinates each
		EEPos = robot_coords(self.bullet_environment["robot"])[EElink-1]
		self.EEPos = EEPos

		state = p.getJointStates(self.bullet_environment["robot"], range(p.getNumJoints(self.bullet_environment["robot"])))
		jointPoses = np.array([s[0] for s in state])

		# Parse keyboard commands.
		EEPos_new = np.copy(EEPos)
		keys = p.getKeyboardEvents()
		left_key = 0
		right_key = 0
		self.num_key_presses += len(keys)
		if p.B3G_UP_ARROW in keys:
			EEPos_new[2] += dist_step[2]
		if p.B3G_DOWN_ARROW in keys:
			EEPos_new[2] -= dist_step[2]
		if p.B3G_LEFT_ARROW in keys:
			EEPos_new[0] -= dist_step[0]
			left_key = 1
		if p.B3G_RIGHT_ARROW in keys:
			EEPos_new[0] += dist_step[0]
			right_key = 1


		# Get new velocity.
		if not np.array_equal(EEPos_new, EEPos):
			newPoses = np.asarray((0.0,) + p.calculateInverseKinematics(self.bullet_environment["robot"], EElink, EEPos_new))
			#print(len(newPoses))
			jointVelocities = (newPoses - jointPoses) / time_step

		if ord('u') in keys:
			jointVelocities[6] -= turn_step / time_step
		if ord('j') in keys:
			jointVelocities[6] += turn_step / time_step

		# Update joystick command.
		self.joy_cmd = np.diag(jointVelocities[1:11])

		if not (self.inference_method == "collect"):
			# Move arm in openrave as well.
			joint_angles = jointPoses[1:8] * (180/np.pi)
			self.joint_angles_callback(joint_angles)
		else:
			# THIS IS THE ONE THAT IS HAPPENING
			self.cmd = self.joy_cmd

		# Update sim position with new velocity command.
		for i in range(len(self.cmd)):
			p.setJointMotorControl2(self.bullet_environment["robot"], i+1, p.VELOCITY_CONTROL, targetVelocity=self.cmd[i][i])

		EEPos = robot_coords(self.bullet_environment["robot"])[EElink-1]
		all_cup_coords = cup_coords(self.bullet_environment["mug"])	
		#print('x distance: ',abs(EEPos[0] - all_cup_coords[0]))
		#print('y distance: ',abs(EEPos[1] - all_cup_coords[1]))
		#print('z distance: ',abs(EEPos[2] - all_cup_coords[2]))

		if (abs(EEPos[0] - all_cup_coords[0]) < x_contact_threshold) and (abs(EEPos[1] - all_cup_coords[1]) < y_contact_threshold) and (abs(EEPos[2] - all_cup_coords[2]) < z_contact_threshold):
			print('made contact, switching to state 4')
			next_state = 4

		return next_state

	def keyboard_input_callback_state4(self):
		# Reset variables.
		jointVelocities = [0.0] * p.getNumJoints(self.bullet_environment["robot"])
		dist_step = [0.0025, 0.0025, 0.0025]
		time_step = 0.05
		turn_step = 0.025
		EElink = 7
		next_state = 4
		object_width = 0.02
		finger_length = 0.06
		distance_between_fingers = 0.038
		# Get current EE position. 
		#robot_coords only gives the 7 degrees of freedom, EEpos is only the position of the end effector. 

		all_coords = robot_coords(self.bullet_environment["robot"])
		#print(all_coords) 
		#there are 11 with 3 coordinates each
		EEPos = robot_coords(self.bullet_environment["robot"])[EElink-1]
		self.EEPos = EEPos

		state = p.getJointStates(self.bullet_environment["robot"], range(p.getNumJoints(self.bullet_environment["robot"])))
		jointPoses = np.array([s[0] for s in state])

		# Parse keyboard commands.
		EEPos_new = np.copy(EEPos)
		keys = p.getKeyboardEvents()
		self.num_key_presses += len(keys)
		if p.B3G_UP_ARROW in keys:
			EEPos_new[2] += dist_step[2]
		if p.B3G_DOWN_ARROW in keys:
			EEPos_new[2] -= dist_step[2]

		# Get new velocity.
		if not np.array_equal(EEPos_new, EEPos):
			newPoses = np.asarray((0.0,) + p.calculateInverseKinematics(self.bullet_environment["robot"], EElink, EEPos_new))
			#print(len(newPoses))
			jointVelocities = (newPoses - jointPoses) / time_step
		if p.B3G_LEFT_ARROW in keys:
			jointVelocities[EElink+2] += turn_step / time_step
			jointVelocities[EElink+3] += turn_step / time_step
			jointVelocities[EElink+1] += turn_step / time_step
		if p.B3G_RIGHT_ARROW in keys:
			jointVelocities[EElink+2] -= turn_step / time_step
			jointVelocities[EElink+3] -= turn_step / time_step
			jointVelocities[EElink+1] -= turn_step / time_step		

		# Update joystick command.
		self.joy_cmd = np.diag(jointVelocities[1:11])

		if not (self.inference_method == "collect"):
			# Move arm in openrave as well.
			joint_angles = jointPoses[1:8] * (180/np.pi)
			self.joint_angles_callback(joint_angles)
		else:
			# THIS IS THE ONE THAT IS HAPPENING
			self.cmd = self.joy_cmd

		# Update sim position with new velocity command.
		for i in range(len(self.cmd)):
			p.setJointMotorControl2(self.bullet_environment["robot"], i+1, p.VELOCITY_CONTROL, targetVelocity=self.cmd[i][i])

		state = p.getJointStates(self.bullet_environment["robot"], range(p.getNumJoints(self.bullet_environment["robot"])))
		jointPoses = np.array([s[0] for s in state])
		distance = math.sin(0.7 - jointPoses[EElink+1])*finger_length*2 + distance_between_fingers

		if (distance < object_width):
			print('grasping, switching to state 5')
			next_state = 5

		return next_state

	def keyboard_input_callback_state5(self):
		# Reset variables.
		jointVelocities = [0.0] * p.getNumJoints(self.bullet_environment["robot"])
		dist_step = [0.0025, 0.0025, 0.0025]
		time_step = 0.05
		turn_step = 0.025
		EElink = 7
		next_state = 5

		# Get current EE position. 
		#robot_coords only gives the 7 degrees of freedom, EEpos is only the position of the end effector. 

		all_coords = robot_coords(self.bullet_environment["robot"])
		#print(all_coords) 
		#there are 11 with 3 coordinates each
		EEPos = robot_coords(self.bullet_environment["robot"])[EElink-1]
		self.EEPos = EEPos

		state = p.getJointStates(self.bullet_environment["robot"], range(p.getNumJoints(self.bullet_environment["robot"])))
		jointPoses = np.array([s[0] for s in state])

		# Parse keyboard commands.
		EEPos_new = np.copy(EEPos)
		keys = p.getKeyboardEvents()
		self.num_key_presses += len(keys)
		if p.B3G_LEFT_ARROW in keys:
			EEPos_new[1] -= dist_step[1]
		if p.B3G_RIGHT_ARROW in keys:
			EEPos_new[1] += dist_step[1]
		if p.B3G_UP_ARROW in keys:
			EEPos_new[0] -= dist_step[0]
		if p.B3G_DOWN_ARROW in keys:
			EEPos_new[0] += dist_step[0]
		if ord('u') in keys:
			EEPos_new[2] += dist_step[2]
		if ord('j') in keys:
			EEPos_new[2] -= dist_step[2]

		# Get new velocity.
		if not np.array_equal(EEPos_new, EEPos):
			newPoses = np.asarray((0.0,) + p.calculateInverseKinematics(self.bullet_environment["robot"], EElink, EEPos_new))
			#print(len(newPoses))
			jointVelocities = (newPoses - jointPoses) / time_step
		if ord('h') in keys:
			jointVelocities[EElink] += turn_step / time_step
		if ord('k') in keys:
			jointVelocities[EElink] -= turn_step / time_step

		# Update joystick command.
		self.joy_cmd = np.diag(jointVelocities[1:11])

		if not (self.inference_method == "collect"):
			# Move arm in openrave as well.
			joint_angles = jointPoses[1:8] * (180/np.pi)
			self.joint_angles_callback(joint_angles)
		else:
			# THIS IS THE ONE THAT IS HAPPENING
			self.cmd = self.joy_cmd

		# Update sim position with new velocity command.
		for i in range(len(self.cmd)):
			p.setJointMotorControl2(self.bullet_environment["robot"], i+1, p.VELOCITY_CONTROL, targetVelocity=self.cmd[i][i])

		return next_state

	def keyboard_input_callback(self):
		# Reset variables.
		jointVelocities = [0.0] * p.getNumJoints(self.bullet_environment["robot"])
		#print(p.getNumJoints(self.bullet_environment["robot"]))
		#dist_step = [0.01, 0.01, 0.01]
		dist_step = [0.0025, 0.0025, 0.0025]
		time_step = 0.05
		turn_step = 0.025
		EElink = 7
		distance_margin = 0.25
		finger_length = 0.04
		distance_between_fingers = 0.041
		# Get current EE position. 

		#robot_coords only gives the 7 degrees of freedom, EEpos is only the position of the end effector. Should see what the remaining 3 degrees are maybe the fingers? See how the wrist is specifically addressed and try to set velocity for fingers too

		all_coords = robot_coords(self.bullet_environment["robot"])
		#print(self.bullet_environment["mug"])
		all_cup_coords = cup_coords(self.bullet_environment["mug"])
		#print(all_cup_coords) 
		#there are 11 with 3 coordinates each
		EEPos = robot_coords(self.bullet_environment["robot"])[EElink-1]
		self.EEPos = EEPos

		#for i, goal_loc in enumerate(self.goal_locs):
		#   print 'distance from goal', i, np.linalg.norm(goal_loc - self.EEPos)

		state = p.getJointStates(self.bullet_environment["robot"], range(p.getNumJoints(self.bullet_environment["robot"])))
		jointPoses = np.array([s[0] for s in state])

		# Parse keyboard commands.
		EEPos_new = np.copy(EEPos)
		keys = p.getKeyboardEvents()
		self.num_key_presses += len(keys)
		if p.B3G_LEFT_ARROW in keys:
			EEPos_new[1] -= dist_step[1]
		if p.B3G_RIGHT_ARROW in keys:
			EEPos_new[1] += dist_step[1]
		if p.B3G_UP_ARROW in keys:
			EEPos_new[0] -= dist_step[0]
		if p.B3G_DOWN_ARROW in keys:
			EEPos_new[0] += dist_step[0]
		if ord('u') in keys:
			EEPos_new[2] += dist_step[2]
		if ord('j') in keys:
			EEPos_new[2] -= dist_step[2]

		#print(EEPos)
		#print(EEPos_new)

		# Get new velocity.
		if not np.array_equal(EEPos_new, EEPos):
			newPoses = np.asarray((0.0,) + p.calculateInverseKinematics(self.bullet_environment["robot"], EElink, EEPos_new))
			#print(len(newPoses))
			jointVelocities = (newPoses - jointPoses) / time_step
		if ord('h') in keys:
			jointVelocities[EElink] += turn_step / time_step
		if ord('k') in keys:
			jointVelocities[EElink] -= turn_step / time_step
		if ord('c') in keys:
			jointVelocities[EElink+1] += turn_step / time_step
		if ord('x') in keys:
			jointVelocities[EElink+1] -= turn_step / time_step
		if ord('d') in keys:
			jointVelocities[EElink+2] += turn_step / time_step
		if ord('f') in keys:
			jointVelocities[EElink+2] -= turn_step / time_step		
		if ord('a') in keys:
			jointVelocities[EElink+3] += turn_step / time_step
		if ord('s') in keys:
			jointVelocities[EElink+3] -= turn_step / time_step

		#joint control instead of inverse kinematics
		# if p.B3G_LEFT_ARROW in keys:
		# 	jointVelocities[EElink-7] -= turn_step / time_step
		# if p.B3G_RIGHT_ARROW in keys:
		# 	jointVelocities[EElink-7] += turn_step / time_step
		# if p.B3G_UP_ARROW in keys:
		# 	jointVelocities[EElink-5] -= turn_step / time_step
		# if p.B3G_DOWN_ARROW in keys:
		# 	jointVelocities[EElink-5] += turn_step / time_step
		# if ord('u') in keys:
		# 	jointVelocities[EElink-6] += turn_step / time_step
		# if ord('j') in keys:
		# 	jointVelocities[EElink-6] -= turn_step / time_step

		# Update joystick command.
		self.joy_cmd = np.diag(jointVelocities[1:11])
		#print 'norm(velocity * timestep) ** 2:', np.linalg.norm(np.array(jointVelocities[1:8]) * self.timestep) ** 2

		#print 'current angles:', jointPoses[1:8] * (180/np.pi)

		if not (self.inference_method == "collect"):
			# Move arm in openrave as well.
			joint_angles = jointPoses[1:8] * (180/np.pi)
			#print(self.inference_method)
			#joint_angles = np.diag(jointPoses[1:8] * (180/np.pi))
			self.joint_angles_callback(joint_angles)
			#print("not collect ")
		else:
			# THIS IS THE ONE THAT IS HAPPENING
			self.cmd = self.joy_cmd
			#print("collect")
			#print(self.inference_method)

		# Update sim position with new velocity command.
		for i in range(len(self.cmd)):
			p.setJointMotorControl2(self.bullet_environment["robot"], i+1, p.VELOCITY_CONTROL, targetVelocity=self.cmd[i][i])

		state = p.getJointStates(self.bullet_environment["robot"], range(p.getNumJoints(self.bullet_environment["robot"])))
		jointPoses = np.array([s[0] for s in state])
		distance = math.sin(0.7 - jointPoses[EElink+1])*finger_length*2 + distance_between_fingers
		print(distance)
		#print(jointPoses[EElink+1])
		#print(jointPoses[EElink+2])
		#print(jointPoses[EElink+3])


	def idx_to_time(self, idx):
		return self.start_T + idx * self.timestep

	def beta_arbitration(self, beta, belief, goal):
		if self.alpha_method == 'joystick':
			return 1.
		elif self.alpha_method == 'zero':
			return 0.
		elif self.alpha_method == 'prob':
			return 1 - belief
		elif self.alpha_method == 'beta':
			#return 1 #all joystick
			#return 0 #all assistance
			#return np.clip(1 / beta, 0, 1)
			return np.clip(1 / beta, 0, 1)
			#return np.clip(0.5 / beta, 0, 1)
			#return np.clip(np.exp(-beta + 0.1), 0, 1)
		elif self.alpha_method == 'dist':
			#return 1
			D = 0.55
			goal_dist = np.linalg.norm(self.goal_locs[goal] - self.EEPos)
			return np.clip(goal_dist / D, 0, 1)


	def update_IK_goals(self):
		self.IK_goals = [p.calculateInverseKinematics(self.bullet_environment["robot"], 7, self.goal_locs[i])[:7] for i in range(self.num_goals)]

if __name__ == '__main__':
	task = int(sys.argv[1])
	method = sys.argv[2]
	TeleopInference(CONFIG_FILE_DICT[task][method])
