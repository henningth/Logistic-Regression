# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 19:51:34 2017

@author: Henning Thomsen

Implements Logistic Regression from exercise 1 in problem set 1 from CS229 at Stanford University.
"""

# Imports necessary libraries
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Parameters
eps = 1e-6 # Threshold

# Loads data from the files
xVec = np.loadtxt("logistic_x.txt") # Points in the (x,y)-plane
yVec = np.loadtxt("logistic_y.txt") # Labels (-1 or 1) of the above points

# Partition data
posVec = xVec[yVec == 1] # Positive examples
negVec = xVec[yVec == -1] # Negative examples

# Includes the intercept term
onesVec = np.ones([np.size(xVec,0),1])
xVec = np.hstack((onesVec,xVec))

# Defines the hypothesis
def h_theta(x,theta):
    return 1/(1+np.exp(-np.dot(x,theta)))

# Computes the gradient vector
def gradient(x, y, theta):
    grad = np.zeros(np.size(x,1))
    for k in np.arange(np.size(grad,0)): # Loop over parameters
        for i in np.arange(np.size(x,0)): # Loop over training examples
            grad[k] = grad[k] - h_theta(-y[i]*x[i,:], theta)*y[i]*x[i,k]
        grad[k] = grad[k]/np.size(x,0)
    return grad

# Computes the Hessian matrix
def hessian(x, y, theta):
    hess = np.zeros([np.size(x,1), np.size(x,1)])
    for k in np.arange(np.size(hess,0)): # Loop over parameters
        for l in np.arange(np.size(hess,1)): # Loop over parameters
            for i in np.arange(np.size(x,0)): # Loop over training examples
                hess[k,l] = hess[k,l] + h_theta(x[i],theta)*(1-h_theta(x[i],theta))*x[i,k]*x[i,l]
            hess[k,l] = hess[k,l]/np.size(x,0)
    return hess

# Computes the empirical loss function
def lossFun(x, y, theta):
    loss = 0
    for i in np.arange(np.size(x,0)): # Loop over training examples
        loss = loss - np.log(h_theta(y[i]*x[i],theta))
    loss = loss / np.size(x,0)
    return loss

# Run Newton's method
theta_old = np.zeros(np.size(xVec,1))
grad = gradient(xVec, yVec, theta_old)
hess = hessian(xVec, yVec, theta_old)
hessInv = np.linalg.inv(hess)

theta = theta_old - np.dot(hessInv,grad)

lossVec = np.array(lossFun(xVec, yVec, theta))

while np.abs(np.linalg.norm(theta-theta_old)) > eps:
    theta_old = theta
    
    grad = gradient(xVec, yVec, theta_old)
    hess = hessian(xVec, yVec, theta_old)
    hessInv = inv(hess)
    
    theta = theta_old - np.dot(hessInv,grad)
    lossVec = np.append(lossVec, lossFun(xVec, yVec, theta))
    
# Plots loss function
plt.figure(1)
plt.plot(lossVec)
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Loss function')

# For the decision boundary
x = np.arange(0,8)
y = -(theta[0] + theta[1]*x)/theta[2]

# Plots data and decision boundary
plt.figure(2)
plt.scatter(posVec[:,0], posVec[:,1], c='red', label='Positive ex.')
plt.hold
plt.scatter(negVec[:,0], negVec[:,1], c='blue', label='Negative ex.')
plt.hold
plt.plot(x, y)
plt.legend()
plt.title('Data and decision boundary')