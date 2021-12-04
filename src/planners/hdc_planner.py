import numpy as np
import math
import json
import copy
import torch


from utils.openrave_utils import *
from utils.trajectory import Trajectory

import pybullet as p
from utils.environment_utils import *


class HDCPlanner(object):
	"""
	This class plans a trajectory from start to goal with HDC.
	"""
	def __init__(self, max_iter, num_waypts, environment, pb_environment, prefer_angles=True, use_constraint_learned=True):

		# ---- Important internal variables ---- #
		# These variables are trajopt parameters.
		self.MAX_ITER = max_iter
		self.num_waypts = num_waypts

		# Set OpenRAVE environment.
		self.environment = environment

		# Set pybullet environment.
		self.bullet_environment = pb_environment

		# whether to use goal angles over goal pose for planning
		self.prefer_angles = prefer_angles

		# whether to use constraints when there are learned features
		self.use_constraint_learned = use_constraint_learned

	# -- Interpolate feature value between neighboring waypoints to help planner optimization. -- #
