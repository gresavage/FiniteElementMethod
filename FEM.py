# coding: utf-8
__author__ = "Arwa Ashi and Tom Gresavage"

import numpy as np
import matplotlib.pyplot as plt
from math import *
from numpy import linalg as LA
from random import *
import math

def utrue(x):
    # Define the true solution
    return np.exp(x**(2./3.))

def g(x, k, v):
    # Define the forcing function of utrue
    return (2.*np.exp(x**(2./3.))*(k - 2.*k*x**(2./3.) + 3.*v*x))/(9.*x**(4./3.))

def linbasis(xj):
    '''
    Creates a linear basis from the 3-element list xj centered at the middle element
    '''
    def f(x):
        if xj[0] <= x <= xj[1]:
            return (x-xj[0])/(xj[1]-xj[0])
        elif xj[1] <= x <= xj[2]:
            return (xj[2]-x)/(xj[2]-xj[1])
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
    Creates a set of linear basis functions centered at each item in xpts
    '''
    dx = np.diff(xpts)
    phi = list()
    for i in range(len(xpts)):
        if i == 0:
            phi.append(linbasis([xpts[i]-dx[i],xpts[i], xpts[i]+dx[i]]))
        elif i == len(xpts)-1:
            phi.append(linbasis([xpts[i]-dx[i-1],xpts[i], xpts[i]+dx[i-1]]))
        else:
            phi.append(linbasis(xpts[i-1:i+2]))
    return phi

# Define the discretization of the problem.
x0 = 1.0      # start of the grid
xn = 10.0   # End of grid
n  = 100       # number of nodes on the grid
dx        = (xn-x0)/float(n) # Set a uniform grid spacing
xpts = np.linspace(x0, xn, n)
dxs = np.diff(xpts) # Calculate \Delta x if mesh is not uniform
phi = genbasis(xpts) # Create basis function

# Define the temporal domain
t0 = 0.
tn = 3.
timeSteps = 100
dt = (tn-t0)/timeSteps

# Set the equation coefficients
k         = 3.0
v         = 1

#set the true solution at time t=0 to be our initial profile.
un = utrue(xpts)
unpo = np.zeros((n,1))
B    = np.zeros((n,n))


# Create the tridiagonal coefficient matrix for the basis functions assuming non-equal spacing
for i in range(1,n-1):
    B[i,i-1] = - v/2. - k/dxs[i-1]
    B[i,i  ] =    k*(1./dxs[i]+1./dxs[i-1])
    B[i,i+1] = + v/2. - k/dxs[i]
B[0,0]     = 1.0
B[-1, -1] = 1.0

# Set the RHS
# Use gaussian quadrature to integrate the right hand side w.r.t x
RHS = [quadrature5(lambda p: phi[i](p)*g(p, k, v), xpts[i-1], xpts[i+1]) for i in range(1, len(xpts)-1)]
#Set the boundary condiations
RHS.insert(0, utrue(xpts[0]))
RHS.append(utrue(xpts[-1]))

# Solve for the coefficients of each basis function
coeff   = LA.solve(B, RHS)

# Update the solution unpo
unpo = np.sum([coeff[i]*np.array(map(phi[i],xpts)) for i in range(len(coeff))], 0)

# Calculate the error
error = np.sum([(quadrature5(lambda p: utrue(p)*phi[i](p), xpts[i-1], xpts[i+1])-quadrature5(lambda p: coeff[i]*phi[i](p), xpts[i-1], xpts[i+1]))**2.*dxs[i-1] for i in range(1,len(xpts)-1)])

plt.plot(xpts, utrue(xpts), 'r')
plt.plot(xpts, unpo)
plt.show()
print sqrt(error)
