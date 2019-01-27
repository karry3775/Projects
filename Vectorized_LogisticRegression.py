#Vectorization of logistic regression

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

#Data
nx = 10 #No of features in the vector
m = 15 #No of Training examples
X = np.random.randint(20, size=(nx,m)).reshape(nx,m)
y = np.random.randint(2, size=(m)).reshape(1,m)

#Helper functions
def sigmoid(z):
    return 1/(1+np.exp(-z))
    
def cost(A,y):
    m = len(y)
    cost = np.dot(y,np.log(A.T)) + np.dot(1-y,np.log(1-A.T))
    cost = -cost/m
    return cost
    

#Initializations
Z = np.zeros_like(y)
W = np.zeros((nx,1))
b = 0
dZ = np.zeros_like(y)
dw = np.zeros_like(W)
db = 0
lr = 0.01 #learning rate
J = np.array([])

#Number of iteration
num_iter = 100

for i in range(num_iter):
    
    #Forward propagation
    Z = np.dot(W.T,X) + b
    A = sigmoid(Z)
    
    #cost function
    J = np.append(J,cost(A,y))
    
    #Backward propagation
    dZ = A - y
    dw = (1/m)*(np.dot(X,dZ.T))
    db = (1/m)*np.sum(dZ)
    
    #Correction step
    W = W - lr*dw
    b = b - lr*db
    
#PLotting the cost function
J = J.reshape(len(J),1)
plt.plot(J)
plt.show()
