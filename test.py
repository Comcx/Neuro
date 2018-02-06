
from numpy.linalg import inv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lainlib as ai


class NeuroLayer:

	layerId = 0

	def __init__(self, transfer, size = None, path = None ):
		if path == None:
			self.__neuroNum = size[0]
			self.__tentacleNum = size[1]
			self.__w = np.random.randn(self.__neuroNum, self.__tentacleNum)
			#self.__w = np.zeros([self.__neuroNum, self.__tentacleNum])
			#self.__b = np.random.randn(self.__neuroNum, 1)
			self.__b = np.zeros([self.__neuroNum, 1])
		else:
			loaded_data = self.load_from(path)
			self.__w = loaded_data[:, :-1].copy()
			self.__b = loaded_data[:, -1:].copy()
			self.__neuroNum = self.__w.shape[0]
			self.__tentacleNum = self.__w.shape[1]

		self.__transfer = transfer

		
	def w(self):
		return self.__w

	def b(self):
		return self.__b

	def z(self, X):
		return np.dot( self.__w, X ) + self.__b

	def reinit_w(self, w):
		self.__w = w

	def reinit_b(self, b):
		self.__b = b

	def shape(self):
		return (self.__neuroNum, self.__tentacleNum)

	def save_to(self, path):
		data = pd.DataFrame(
			np.concatenate([self.__w, self.__b], axis = 1))
		data.to_csv(path)

	def load_from(self, path):
		data = pd.read_csv(path).values[:, 1:]
		return data

	def act(self, X):
		Z = self.z(X)
		return self.__transfer(Z)

	def lost(self, A, Y):
		return np.power((Y - A), 2).sum()

	def hebb_learn_init_with(self, X, Y):
		self.__w = np.dot(Y, X.T)

	def hebb_learn(self, X, a = 1):
		A = self.act(X)
		#print(np.dot(Z, X.T))
		self.__w = self.__w + a * np.dot(A, X.T)

	def penrose_pseudoinverse_learn(self, X, Y):
		X_plus = np.dot(inv(np.dot(X.T, X)), X.T)
		self.__w = np.dot(Y, X_plus)

# -> end of class NeuroLayer



def hard_limits(Z):
	return ai.hard_limit(Z, 0, [-1, 1])


X = np.array([
	[1, 0, 0],
	[1, 1, 0],
	[1, 1, 1]
	]).T

Y = np.array([
	[0, 1, 0],
	[1, 0, 0],
	[1, 1, 1]
	]).T


test = NeuroLayer(hard_limits, [3, 3])
#test.penrose_pseudoinverse_learn(hard_limits(X), hard_limits(Y))
#test.hebb_learn_init_with(hard_limits(X), hard_limits(Y))

for i in range(1000):
	test.hebb_learn(hard_limits(X), 1 / (i+1))
	A = test.act(hard_limits(X))
	lost = test.lost(A, hard_limits(Y))

	if lost > 1:
		test.reinit_w(np.random.randn(3, 3))


print(test.w())
A = test.act(hard_limits(X))
print(ai.hard_limit(A, 0, [0, 1]))
print(test.lost(A, hard_limits(Y)))
#test.save_to("C:/Users/HP/Desktop/temp.csv")











'''
plt.figure(figsize = (2,4))
x = [0, 0, 0, 1, 1, 1, 0]
y = [0, 1, 2, 2, 1, 0, 0]
plt.plot(x, y, 'r-', linewidth = 10)
plt.plot([0, 1], [1, 1], 'r-', linewidth = 10)

#plt.grid(True)
#plt.legend()
#plt.show()
'''















