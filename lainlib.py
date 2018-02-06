
import numpy as np

def hard_limit(Z, divide_value, range):
	return np.where(Z <= divide_value, range[0], range[1])

def log_sigmoid(Z):
	return 1 / (1 + np.exp(-Z))



