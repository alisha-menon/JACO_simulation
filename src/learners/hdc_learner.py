import numpy as np
import os
import itertools
import pickle
import matplotlib.pyplot as plt
import matplotlib
import ast

class HDCLearner(object):
	"""
	This class creates a Program HV that represents all the learned sensor + actuator pairs given human inputs.
	"""

	def __init__(self, feat_list, environment, constants):
		# ---- Important internal variables ---- #
		self.feat_list = feat_list
		self.num_features = len(self.feat_list)
		self.environment = environment
