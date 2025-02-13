#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:51:51 2025

@author: ehansen
"""


import numpy as np
from scipy.linalg import solve_banded,solve
from scipy.special import roots_legendre
import matplotlib.pyplot as plt


sigmat = 1.0
sigmas = 0.8
length = 100
Nx = 10
dx = length/Nx
Nmu = 2
q0 = 1

mus,weight = roots_legendre(2)

# Use Bell and Glasstone equation, and use average of moments at edges
# for value within the cell

# Also use equations from lecture
# (phi1(i+1)-phi1(i))/dx + sigmat (phi0(i+1)+phi0(i))/2 = q0 + sigmas * phi0
# 1/3 * (phi0(i+1)-phi0(i))/dx + sigmat (phi1(i+1)+phi1(i))/2 = 0
 
psi = np.zeros([Nmu*(Nx+1)])

matrix = np.zeros([Nmu*(Nx+1),Nmu*(Nx+1)])


b = psi.copy()
b2 = psi.copy()
x = psi.copy()
y = psi.copy()

# Reflecting boundary conditions for starting point
matrix[0,1] = 1
b[0] = 0

# Equations - take dx to infinity limit for now
# My first attempt was singular without the derivative terms

for i in range(Nx):
    matrix[1+Nmu*i,i*Nmu] = sigmat/2
    matrix[1+Nmu*i,i*Nmu+1] = -1/dx
    matrix[1+Nmu*i,i*Nmu+2] = sigmat/2
    matrix[1+Nmu*i,i*Nmu+3] = 1/dx
    
    b[1+Nmu*i] = q0
    
    matrix[2+Nmu*i,i*Nmu] = -1/(3*dx)
    matrix[2+Nmu*i,i*Nmu+1] = sigmat/2
    matrix[2+Nmu*i,i*Nmu+2] = 1/(3*dx)
    matrix[2+Nmu*i,i*Nmu+3] = sigmat/2
    b[2+Nmu*i] = 0

# Last equation - reflecting boundary condition at end
matrix[-1,-1] = 1
b[-1] = 0

scalarflux = np.zeros(Nx)
oldflux = scalarflux.copy()+10

def update_scalarflux(x):
    # Just need the scalar flux
    scalarflux = (x[0:-2:Nmu]+x[2::Nmu])/2
    return(scalarflux)

print(matrix)

i = 0
while np.amax(np.abs(scalarflux-oldflux)) > 10**(-10):
    oldflux = scalarflux
    x = solve(matrix,b)
    
    scalarflux = update_scalarflux(x)
    b[1:-1:2] = q0 + sigmas*scalarflux
    b[2:-1:2] = 0
    
    print("Iteration ",i," Difference ", np.amax(np.abs(scalarflux-oldflux)))
    i += 1 
    
plt.plot(np.arange(0,Nx+1)*length/Nx,x[0::2],"r",label="Scalar Flux")
plt.plot(np.arange(0,Nx+1)*length/Nx,x[1::2],"b",label="Current")

plt.xlabel("x position (cm)")
plt.title("Moments of Angular Flux")
if np.amax(scalarflux < 3):
    plt.ylim(-3,3)
plt.ylabel("Moment")
plt.legend()
plt.show()

# Repeat the problem but banded

b[0] = 0
b[-1] = 0
b[Nmu//2:-Nmu//2:Nmu ] = q0

AB = np.zeros([Nmu+3,(Nx+1)*Nmu])

# Reflecting boundary conditions - set current to zero
AB[Nmu-1,1] = 1
AB[Nmu,-1] = 1

# Set up equations - read from printed array above
for i in range(Nx):
    print(1+(i+1)*Nmu,1+(i+2)*Nmu)
    AB[0,2+2*i:2+2*(i+1)] = np.array([0,1/dx])
    AB[1,2+2*i:2+2*(i+1)] = np.array([sigmat/2,sigmat/2])
    AB[2,1+2*i:1+2*(i+1)] = np.array([-1/dx,1/(3*dx)])
    AB[3,i*Nmu:(i+1)*Nmu] = np.array([sigmat/2,sigmat/2])
    AB[4,i*Nmu:(i+1)*Nmu] = np.array([-1/(3*dx),0])

print(AB)

scalarflux = 0.0*scalarflux
oldflux = scalarflux.copy()-99
while np.amax(np.abs(scalarflux-oldflux)) > 10**(-10):
    oldflux = scalarflux
    x = solve_banded((2,2),AB,b)
    
    scalarflux = update_scalarflux(x)
    b[1:-1:2] = q0 + sigmas*scalarflux
    b[2:-1:2] = 0
    
    print("Iteration ",i," Difference ", np.amax(np.abs(scalarflux-oldflux)))
    i += 1 

plt.plot(np.arange(0,Nx+1)*length/Nx,x[0::2],"r",label="Scalar Flux")
plt.plot(np.arange(0,Nx+1)*length/Nx,x[1::2],"b",label="Current")

plt.xlabel("x position (cm)")
plt.title("Moments of Angular Flux")
if np.amax(scalarflux < 3):
    plt.ylim(-3,3)
plt.ylabel("Moment")
plt.legend()
plt.show()
