import numpy as np
import time
import os
import pickle

from utils.openrave_utils import *

class ExperimentUtils(object):

	def __init__(self, save_dir):
		# Stores dictionary of all the replanned trajectories
		self.replanned_trajList = {}

		# Stores dictionary of all the replanned waypoints
		self.replanned_wayptsList = {}

		# Stores trajectories as they are deformed
		self.deformed_trajList = {}

		# Stores trajectory waypoints as they are deformed
		self.deformed_wayptsList = {}

		# Stores the list of positions as the robot is moving 
		# in the form [timestamp, j1, j2, ... , j7]
		self.tracked_traj = None

		# Stores list of interaction waypoints
		self.interaction_pts = None

		# Stores weights over time 
		# always in the form [timestamp, weight1, ..., weightN]
		self.weights = None

		# Stores betas over time 
		# always in the form [timestamp, beta1, ..., betaN]
		self.betas = None

		# Stores betas over time 
		# always in the form [timestamp, beta1, ..., betaN]
		self.betas_u = None

		# Stores updates over time 
		# always in the form [timestamp, update1, ..., updateN]
		self.updates = None

		# Stores running list of forces applied by human
		# in the form [timestamp, j1, j2, ... , j7]
		self.tauH = None 

		# Stores start and end time of the interaction
		self.startT = 0.0
		self.endT = 0.0

		# Store saved directory path
		here = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../'))
		self.save_dir = here + save_dir + "/"

	# ----- Data saving/tracking utils ----- 

	def update_replanned_trajList(self, timestamp, traj):
		"""
		Updates the dictionary of replanned trajectories since the start.
		"""
		if timestamp not in self.replanned_trajList:
			self.replanned_trajList[timestamp] = traj

	def update_replanned_wayptsList(self, timestamp, waypts):
		"""
		Updates the dictionary of replanned trajectory waypoints since the start.
		"""
		if timestamp not in self.replanned_wayptsList:
			self.replanned_wayptsList[timestamp] = waypts

	def update_deformed_trajList(self, timestamp, waypts):
		"""
		Updates the deformed trajectory.
		"""
		if timestamp not in self.deformed_trajList:
			self.deformed_trajList[timestamp] = waypts

	def update_deformed_wayptsList(self, timestamp, waypts):
		"""
		Updates the deformed trajectory waypoints.
		"""
		if timestamp not in self.deformed_wayptsList:
			self.deformed_wayptsList[timestamp] = waypts

	def update_tracked_traj(self, timestamp, curr_pos):
		"""
		Uses current position read from the robot to update the trajectory.
		Saves timestamp when this position was read.
		""" 
		currTraj = np.append([timestamp], curr_pos.reshape(7))
		if self.tracked_traj is None:
			self.tracked_traj = np.array([currTraj])
		else:	
			self.tracked_traj = np.vstack([self.tracked_traj, currTraj])

	def update_tauH(self, timestamp, tau_h):
		"""
		Uses current joint torque reading from the robot during interaction.
		Saves timestamp when this torque was read.
		""" 
		currTau = np.append([timestamp], tau_h.reshape(7))
		if self.tauH is None:
			self.tauH = np.array([currTau])
		else:
			self.tauH = np.vstack([self.tauH, currTau])

	def update_interaction_point(self, timestamp, interaction_point):
		"""
		Uses current position reading from the robot during interaction.
		Saves timestamp when this position was read.
		""" 
		curr_iact_pt = np.append([timestamp], interaction_point.reshape(7))
		if self.interaction_pts is None:
			self.interaction_pts = np.array([curr_iact_pt])
		else:
			self.interaction_pts = np.vstack([self.interaction_pts, curr_iact_pt])

	def update_weights(self, timestamp, new_weight):
		"""
		Updates list of timestamped weights.
		"""
		if new_weight is None:
			print "In update_weights: new_weights are None..."
			return 
		new_w = np.array([timestamp] + new_weight)
		if self.weights is None:
			self.weights = np.array([new_w])
		else:
			self.weights = np.vstack([self.weights, new_w])

	def update_betas(self, timestamp, new_beta):
		"""
		Updates list of timestamped betas.
		"""
		if new_beta is None:
			print "In update_betas: new_beta is None..."
			return 
		new_b = np.array([timestamp] + new_beta)
		if self.betas is None:
			self.betas = np.array([new_b])
		else:
			self.betas = np.vstack([self.betas, new_b])

	def update_betas_u(self, timestamp, new_beta_u):
		"""
		Updates list of timestamped betas_u.
		"""
		if new_beta_u is None:
			print "In update_betas_u: new_beta_u is None..."
			return 
		new_b_u = np.array([timestamp] + new_beta_u)
		if self.betas_u is None:
			self.betas_u = np.array([new_b_u])
		else:
			self.betas_u = np.vstack([self.betas_u, new_b_u])
	
	def update_updates(self, timestamp, new_update):
		"""
		Updates list of timestamped updates.
		"""
		if new_update is None:
			print "In update_updates: new_update is None..."
			return 
		new_u = np.array([timestamp] + new_update)
		if self.updates is None:
			self.updates = np.array([new_u])
		else:
			self.updates = np.vstack([self.updates, new_u])

	def set_startT(self,start_t):
		"""
		Records start time for experiment.
		"""
		self.startT = start_t

	def set_endT(self,end_t):
		"""
		Records end time for experiment.
		"""
		self.endT = end_t

	# ----- Saving (pickling) utilities ------- #

	def get_unique_filepath(self,subdir,filename):
		# Get the current script path
		dirpath = self.save_dir + subdir + "/"
		filepath = dirpath + filename + "_1.p"
		i = 2
		while os.path.exists(filepath):
			filepath = dirpath + filename + str(i)+".p"
			i+=1

		return filepath

	def get_unique_bagpath(self,subdir,filename):
		# Get the current script path.
		dirpath = self.save_dir + subdir + "/"
		filepath = dirpath + filename + "_1.bag"
		i = 2
		while os.path.exists(filepath):
			filepath = dirpath + filename + str(i)+".bag"
			i+=1

		return filepath

	def pickle_replanned_trajList(self, filename):
		"""
		Pickles the replanned_trajList data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("replanned",filename)
		pickle.dump(self.replanned_trajList, open( filepath, "wb" ) )

	def pickle_replanned_wayptsList(self, filename):
		"""
		Pickles the replanned_wayptsList data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("replanned_waypts",filename)
		pickle.dump(self.replanned_wayptsList, open( filepath, "wb" ) )

	def pickle_deformed_trajList(self, filename):
		"""
		Pickles the deformed_trajList data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("deformed",filename)
		pickle.dump(self.deformed_trajList, open( filepath, "wb" ) )

	def pickle_deformed_wayptsList(self, filename):
		"""
		Pickles the deformed_wayptsList data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("deformed_waypts",filename)
		pickle.dump(self.deformed_wayptsList, open( filepath, "wb" ) )

	def pickle_tracked_traj(self, filename):
		"""
		Pickles the tracked_traj data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("tracked",filename)
		pickle.dump(self.tracked_traj, open( filepath, "wb" ) )

	def pickle_weights(self, filename):
		"""
		Pickles the weights data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("weights",filename)
		pickle.dump(self.weights, open( filepath, "wb" ) )

	def pickle_betas(self, filename):
		"""
		Pickles the betas data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("betas",filename)
		pickle.dump(self.betas, open( filepath, "wb" ) )

	def pickle_betas_u(self, filename):
		"""
		Pickles the betas_u data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("betas_u",filename)
		pickle.dump(self.betas_u, open( filepath, "wb" ) )

	def pickle_updates(self, filename):
		"""
		Pickles the update data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("updates",filename)
		pickle.dump(self.updates, open( filepath, "wb" ) )

	def pickle_interaction_pts(self, filename):
		"""
		Pickles the interaction_pts data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("interaction_pts",filename)
		pickle.dump(self.interaction_pts, open( filepath, "wb" ) )

	def pickle_force(self, filename):
		"""
		Pickles the force data structure for later analysis. 
		"""

		filepath = self.get_unique_filepath("force",filename)
		pickle.dump(self.tauH, open( filepath, "wb" ) )
