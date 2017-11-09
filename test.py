# Python for CS exploring
# -*- coding: UTF-8 -*-

import numpy as np

class Neuro:

    w = np.zeros(4)
    b = 0
    ratio = 0.01

    def __init__(self,
                 w = np.zeros((1, 4)), b = 0):
        self.w = w
        self.b = b


    def act( self, X ):
        def threshold(Z):
            return 1 / (1 + np.exp(-Z))
        Z = np.dot( self.w, X ) + self.b

        return threshold(Z)

    def cost(self, y, Y):
        list = -(
            Y * np.log(y) + (1-Y) * np.log(1-y)
        )
        return np.sum(list) / np.shape(y)[0]

    def update(self, X, Y):
        A = self.act(X)
        dz = A - Y
        db = np.sum(dz) / np.shape(X)[1]
        dw = np.dot(X, dz.T) / np.shape(X)[1]
        return Neuro( self.w-self.ratio*dw.T, self.b-self.ratio*db )





test = Neuro()
X = np.array([
    [1,2,3,4],
    [4,5,6,7],
    [7,8,9,0],
    [8,1,4,6],
    [5,8,5,8]
]).T
Y = np.array([0.758, 0.791, 0.099, 0.029, 0.1])


print(test.act(X))
print('-> cost: %s'%(test.cost(test.act(X), Y)))

for i in range(1, 1000):
    print('\n>> Time #%s' % (i))
    test = test.update(X, Y)
    print('-> cost: %s' % (test.cost(test.act(X), Y)))
    print('-> w: %s' % (test.w))


x = np.array([10,2,0,4]).T
print(test.act(X))











