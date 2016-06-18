# coding: utf-8
__author__ = "Arwa Ashi and Tom Gresavage"
import numpy as np
import matplotlib.pyplot as plt
from math import *
from numpy import linalg as LA
from random import *
import math

def utrue(t, x):
    # Define the true solution
    return np.exp(-t)*np.cos(x)*np.sin(x)

def g(t, x, k, v):
    # Define the forcing function of utrue
    return np.exp(-t)*(2.*v*np.cos(2.*x) + (-1. + 4.*k)*np.sin(2.*x))/2.
    # return (2.*np.exp(x**(2./3.))*(k - 2.*k*x**(2./3.) + 3.*v*x))/(9.*x**(4./3.))

def quadbasis(xj):
    '''
    Creates a linear basis from the 3-element list xj centered at the middle element
    '''
    def f(x):
        if xj[0] <= x <= xj[2]:
            return (x*-xj[0])*(xj[2]-x)/((xj[1]-xj[0])*(xj[2]-xj[1]))
        else:
            return 0.
    return f

def quadrature5(f, a, b):
    '''
    Calculate int_a^b f(x)dx using 5-point Gaussian Quadrature
    Print the result
    '''
    quad_x = [0., np.sqrt(5.-2.*np.sqrt(10./7.))/3., -np.sqrt(5.-2.*np.sqrt(10./7.))/3., np.sqrt(5.+2.*np.sqrt(10./7.))/3., -np.sqrt(5.+2.*np.sqrt(10./7.))/3.]
    quad_w = [128./225., (322.+13.*np.sqrt(70))/900., (322.+13.*np.sqrt(70))/900., (322.-13.*np.sqrt(70))/900., (322.-13.*np.sqrt(70))/900.]
    s_int = ((b-a)/2.)*np.sum([quad_w[j]*f(((b-a)/2.)*quad_x[j]+(a+b)/2.) for j in range(5)])
    return s_int

# Define a function to generate a set of basis functions for each x
def genbasis(xpts):
    '''
    Creates a set of basis functions centered at each item in xpts
    '''

    '''
    The basis function (phi) must further discretize the space between each x (dxpts)
    This is because the solution will describe the space between each x
    '''
    dx = np.diff(xpts)
    dxpts = np.linspace(xpts[0], xpts[-1], 100)
    phi = list()
    y = list()
    for i in range(len(xpts)):
        if i == 0:
            phi.append(quadbasis([xpts[i]-dx[i],xpts[i], xpts[i]+dx[i]]))
        elif i == len(xpts)-1:
            phi.append(quadbasis([xpts[i]-dx[i-1],xpts[i], xpts[i]+dx[i-1]]))
        else:
            phi.append(quadbasis(xpts[i-1:i+2]))
        y.append(map(phi[i], dxpts))
    return dxpts, y, phi

# Define the discretization of the problem.
x0 = 1.0      # start of the grid
xn = 10.0   # End of grid
n  = 20       # number of nodes on the grid
dx = (xn-x0)/float(n) # Set a uniform grid spacing
xpts = np.linspace(x0, xn, n)
dxs = np.diff(xpts) # Calculate \Delta x if mesh is not uniform
t, y, phi = genbasis(xpts) # Create basis function

# Define the temporal domain
t0 = 0.
tn = 3.
timeSteps = 10
dt = (tn-t0)/timeSteps
saveEvery = 2

# Set the equation coefficients
k         = 3.0
v         = 1.

# Convenience Coefficients
alfa      = v/2.0
gamma     = k/dx

# Create variables to save the error at each iteration
Error = []
error = 0.0

# Set the true solution at time t=0 to be our initial profile.
coeff = utrue(t0, xpts)
unpo = np.zeros((n,1))
B    = np.zeros((n,n))


# Create the tridiagonal coefficient matrix for the basis functions assuming non-equal spacing
for i in range(1,n-1):
    B[i,i-1] = (- v/6. - 2.*k/(3.*dxs[i-1]))*dt + dxs[i-1]/30.
    B[i,i  ] =    4.*k*dt*(1./dxs[i] + 1./dxs[i-1])/3. + (dxs[i-1] - dxs[i])/5.
    B[i,i+1] = (+ v/6. + 2.*k/(3.*dxs[i]))*dt + dxs[i]/30.
B[0,0]     = 1.0
B[-1, -1] = 1.0

for b in range(timeSteps):
    # Set the iteration time
    tIter = t0+dt*b

    # Set the RHS
    # Use gaussian quadrature to integrate the right hand side w.r.t. x
    RHS = [dxs[i-1]*coeff[i-1]/30.+(dxs[i-1] - dxs[i])*coeff[i]/5.+dxs[i]*coeff[i+1]/30.+dt*quadrature5(lambda p: phi[i](p)*g(tIter, p, k, v), xpts[i-1], xpts[i+1]) for i in range(1, len(xpts)-1)]
    #Set the boundary condiations
    RHS.insert(0, utrue(tIter, xpts[0]))
    RHS.append(utrue(tIter, xpts[-1]))

    # Solve for the coefficients of each basis function
    coeff   = LA.solve(B, RHS)

    # Update the solution unpo
    unpo = np.sum([coeff[i]*np.array(y[i]) for i in range(len(coeff))], 0)

    # Calculate the iteration error
    stepError = np.sum([(quadrature5(lambda p: utrue(t,p), xpts[i-1], xpts[i+1])-quadrature5()*dx
    error     = error + stepError*dt

    if (b%saveEvery==0):
        Error.append(error**0.5)
        plt.plot(xpts, utrue(tIter, xpts), 'r')
        plt.plot(t, unpo, 'b')
print Error
plt.show()